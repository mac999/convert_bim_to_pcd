"""
texture_manager.py
==================
IFC product 클래스(카테고리)별로 어울리는 재질 텍스처를 확보하는 모듈.

우선순위:
  1) ``textures/<category>.jpg`` 가 이미 있으면 그대로 사용 (캐시)
  2) ambientCG(https://ambientcg.com) 의 CC0 재질 라이브러리에서
     카테고리별 검색어(texture_query)로 검색 → 1K-JPG 컬러맵 다운로드
  3) 네트워크 실패 시 numpy/PIL 로 절차적(procedural) 텍스처 생성 (오프라인 폴백)

ambientCG 의 모든 에셋은 CC0(퍼블릭 도메인) 라이선스이므로 자유롭게 사용/재배포 가능.
다운로드 이력과 출처/라이선스는 ``textures/manifest.json`` 에 기록한다.
"""

import io
import ssl
import json
import zipfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

AMBIENTCG_API = "https://ambientcg.com/api/v2/full_json"
_SSL_CTX = ssl.create_default_context()
# 일부 윈도우 환경의 인증서 체인 문제를 우회 (CC0 이미지 다운로드 용도)
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# config 최상위의 예약 키(카테고리가 아님)
RESERVED_KEYS = {"settings"}

# distinct 모드 기본 팔레트 (config 에 distinct_color 가 없을 때 폴백)
# 클래스를 시각적으로 확실히 구분하도록 채도 높은 색을 배정
DEFAULT_DISTINCT = {
    "wall": (205, 200, 190), "column": (0, 200, 210), "beam": (255, 140, 20),
    "slab": (120, 200, 90), "roof": (150, 58, 42), "door": (160, 95, 45),
    "window": (35, 90, 235), "stair": (175, 80, 205), "railing": (235, 45, 150),
    "covering": (0, 175, 165), "furniture": (245, 205, 40),
}
# distinct_color/DEFAULT_DISTINCT 둘 다 없을 때 인덱스로 뽑는 채도 높은 색환
_DISTINCT_WHEEL = [
    (230, 40, 35), (35, 90, 235), (40, 190, 70), (245, 205, 40), (175, 80, 205),
    (0, 195, 205), (255, 140, 20), (235, 45, 150), (0, 175, 120), (150, 95, 45),
]

# 절차적 폴백에 사용할 대표 색상 (검색어 키워드 → 베이스 컬러/패턴)
_PROC_STYLES = {
    "brick": ((150, 70, 55), "brick"),
    "plaster": ((205, 200, 190), "noise"),
    "concrete": ((175, 175, 178), "noise"),
    "wood": ((150, 100, 60), "wood"),
    "plank": ((150, 100, 60), "wood"),
    "metal": ((160, 162, 168), "metal"),
    "steel": ((150, 152, 160), "metal"),
    "glass": ((150, 185, 205), "glass"),
    "tile": ((170, 120, 95), "brick"),
    "roof": ((140, 85, 65), "brick"),
    "marble": ((225, 222, 218), "noise"),
}


def _http_get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ifc2las texture fetch)"})
    return urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX).read()


def _find_ambientcg_by_id(asset_id, resolution="1K-JPG"):
    """ambientCG 에서 정확한 assetId 로 (asset_id, color_zip_url) 반환. 실패 시 None.
    config 의 texture_query 검색이 원하는 재질을 못 집을 때 특정 에셋을 고정 지정하는 용도."""
    # ambientCG 는 정확한 에셋 조회에 q= 가 아니라 id= 파라미터를 사용
    params = urllib.parse.urlencode({"id": asset_id, "include": "downloadData"})
    data = json.loads(_http_get(f"{AMBIENTCG_API}?{params}", timeout=30))
    for asset in data.get("foundAssets", []):
        if asset.get("assetId") != asset_id:
            continue
        cats = asset.get("downloadFolders", {}).get("default", {}).get(
            "downloadFiletypeCategories", {})
        for info in cats.values():
            for dl in info.get("downloads", []):
                if dl.get("attribute") == resolution:
                    return asset_id, dl.get("downloadLink")
    return None


def _find_ambientcg_download(query, resolution="1K-JPG"):
    """ambientCG 에서 query 로 재질 검색 → (asset_id, color_zip_url) 반환. 실패 시 None."""
    params = urllib.parse.urlencode({
        "type": "Material",
        "limit": "1",
        "q": query,
        "include": "downloadData",
    })
    url = f"{AMBIENTCG_API}?{params}"
    data = json.loads(_http_get(url, timeout=30))
    assets = data.get("foundAssets", [])
    if not assets:
        return None
    asset = assets[0]
    asset_id = asset.get("assetId")
    folders = asset.get("downloadFolders", {})
    cats = folders.get("default", {}).get("downloadFiletypeCategories", {})
    for info in cats.values():
        for dl in info.get("downloads", []):
            if dl.get("attribute") == resolution:
                return asset_id, dl.get("downloadLink")
    return None


def _extract_color_jpg(zip_bytes, out_path):
    """ambientCG zip 에서 *_Color.* 맵을 찾아 out_path(jpg) 로 저장."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    color_name = None
    for name in zf.namelist():
        low = name.lower()
        if "_color." in low and low.rsplit(".", 1)[-1] in ("jpg", "jpeg", "png"):
            color_name = name
            break
    if color_name is None:
        return False
    img = Image.open(io.BytesIO(zf.read(color_name))).convert("RGB")
    img.save(out_path, quality=90)
    return True


# ---------------------------------------------------------------------------
# 절차적 텍스처 (오프라인 폴백)
# ---------------------------------------------------------------------------
def _procedural_texture(base_color, pattern, size=512, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array(base_color, dtype=np.float32)
    img = np.tile(base, (size, size, 1))

    if pattern == "brick":
        bh, bw = size // 8, size // 4
        mortar = np.array([210, 205, 198], dtype=np.float32)
        for row in range(0, size, bh):
            offset = (bw // 2) if (row // bh) % 2 else 0
            img[row:row + 2, :] = mortar  # 가로 줄눈
            for col in range(-offset, size, bw):
                c = max(col, 0)
                img[row:row + bh, c:c + 2] = mortar  # 세로 줄눈
        img += rng.normal(0, 10, (size, size, 1))
    elif pattern == "wood":
        x = np.linspace(0, 22 * np.pi, size)
        grain = (np.sin(x) * 0.5 + 0.5)[None, :, None]
        grain = grain + rng.normal(0, 0.06, (size, size, 1))
        img = base * (0.72 + 0.35 * grain)
    elif pattern == "metal":
        streak = rng.normal(0, 6, (1, size, 1)).repeat(size, axis=0)
        img += streak + rng.normal(0, 3, (size, size, 1))
    elif pattern == "glass":
        grad = np.linspace(-18, 18, size)[:, None, None]
        img += grad + rng.normal(0, 4, (size, size, 1))
    else:  # noise (plaster / concrete / marble)
        fine = rng.normal(0, 8, (size, size, 1))
        coarse = rng.normal(0, 14, (size // 16, size // 16, 1))
        coarse = np.array(Image.fromarray(
            np.clip(coarse + 128, 0, 255).astype(np.uint8)[:, :, 0]
        ).resize((size, size))).astype(np.float32)[:, :, None] - 128
        img += fine + coarse

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _pick_proc_style(query):
    q = query.lower()
    for key, style in _PROC_STYLES.items():
        if key in q:
            return style
    return ((180, 180, 180), "noise")


# ---------------------------------------------------------------------------
# distinct 텍스처 (클래스 구분용 고채도 색 + 약한 재질 패턴)
# ---------------------------------------------------------------------------
def _distinct_texture(base_color, pattern, size=512, seed=0):
    """
    고채도 베이스 색을 유지하면서 재질 느낌(패턴)을 약하게만 얹은 텍스처.
    점군 색상으로 샘플링했을 때 클래스별 색이 확실히 구분되도록 색조를 지배시킨다.
    """
    rng = np.random.default_rng(seed)
    base = np.array(base_color, dtype=np.float32)
    img = np.tile(base, (size, size, 1))

    if pattern == "brick" or pattern == "tile":
        bh, bw = size // 8, size // 4
        line = base * 0.68
        for row in range(0, size, bh):
            offset = (bw // 2) if (row // bh) % 2 else 0
            img[row:row + 3, :] = line
            for col in range(-offset, size, bw):
                c = max(col, 0)
                img[row:row + bh, c:c + 3] = line
        img += rng.normal(0, 6, (size, size, 1))
    elif pattern == "wood":
        x = np.linspace(0, 22 * np.pi, size)
        grain = (np.sin(x) * 0.5 + 0.5)[None, :, None]
        img = base * (0.82 + 0.24 * grain) + rng.normal(0, 5, (size, size, 1))
    elif pattern == "metal":
        streak = rng.normal(0, 7, (1, size, 1)).repeat(size, axis=0)
        img += streak + rng.normal(0, 3, (size, size, 1))
    elif pattern == "glass":
        grad = np.linspace(-16, 16, size)[:, None, None]
        img += grad + rng.normal(0, 4, (size, size, 1))
    else:  # noise (plaster / concrete / marble)
        img += rng.normal(0, 8, (size, size, 1))

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_distinct_textures(config, categories, out_dir, distinct_colors=None,
                           size=512, force=False, verbose=True):
    """
    카테고리별 고채도 distinct 텍스처를 out_dir 에 생성하고 {category: Path} 반환.
    색상 우선순위: config 의 distinct_color > DEFAULT_DISTINCT > 색환(인덱스).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distinct_colors = distinct_colors or {}
    result = {}
    for i, category in enumerate(categories):
        data = config.get(category, {})
        query = data.get("texture_query", category) if isinstance(data, dict) else category
        _, pattern = _pick_proc_style(query)
        color = (distinct_colors.get(category)
                 or DEFAULT_DISTINCT.get(category)
                 or _DISTINCT_WHEEL[i % len(_DISTINCT_WHEEL)])
        out_path = out_dir / f"{category}.jpg"
        if out_path.exists() and not force:
            if verbose:
                print(f"  [distinct] {category:<10} cached  -> {out_path.name}")
        else:
            seed = abs(hash(category)) % (2 ** 31)
            _distinct_texture(color, pattern, size=size, seed=seed).save(out_path, quality=92)
            if verbose:
                print(f"  [distinct] {category:<10} generated rgb{tuple(color)} ({pattern})")
        result[category] = out_path
    return result


def _make_procedural(category, query, out_path):
    color, pattern = _pick_proc_style(query)
    seed = abs(hash(category)) % (2 ** 31)
    _procedural_texture(color, pattern, seed=seed).save(out_path, quality=90)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def ensure_textures(config, textures_dir, allow_download=True, resolution="1K-JPG",
                    verbose=True, mode="realistic", categories=None, distinct_colors=None,
                    force_distinct=False):
    """
    각 카테고리에 대해 텍스처 이미지를 확보하고 {category: Path(jpg)} 매핑을 반환한다.

    mode == "distinct" 이면 ambientCG/절차적 대신 클래스 구분용 고채도 텍스처를
    ``textures_dir/distinct`` 에 생성해 사용한다.
    """
    textures_dir = Path(textures_dir)
    textures_dir.mkdir(parents=True, exist_ok=True)
    if categories is None:
        categories = [k for k in config.keys() if k not in RESERVED_KEYS]

    if mode == "distinct":
        return make_distinct_textures(config, categories, textures_dir / "distinct",
                                      distinct_colors=distinct_colors,
                                      force=force_distinct, verbose=verbose)

    manifest_path = textures_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    result = {}
    for category in categories:
        data = config.get(category, {})
        out_path = textures_dir / f"{category}.jpg"
        query = data.get("texture_query", category) if isinstance(data, dict) else category

        if out_path.exists():
            result[category] = out_path
            if verbose:
                print(f"  [texture] {category:<10} cached  -> {out_path.name}")
            continue

        # config 에 texture_asset(정확한 ambientCG 에셋 ID)가 있으면 검색 대신 그 에셋을 고정 사용
        pinned = data.get("texture_asset") if isinstance(data, dict) else None
        # procedural_only: skip download and always generate (e.g. water — no CC0 photo material fits)
        proc_only = bool(data.get("procedural_only")) if isinstance(data, dict) else False

        source = None
        if allow_download and not proc_only:
            try:
                found = None
                if pinned:
                    found = (_find_ambientcg_by_id(pinned, resolution)
                             or _find_ambientcg_download(pinned, resolution=resolution))
                if not found:
                    found = _find_ambientcg_download(query, resolution=resolution)
                if found:
                    asset_id, link = found
                    if _extract_color_jpg(_http_get(link), out_path):
                        source = {"provider": "ambientCG", "asset": asset_id,
                                  "license": "CC0", "url": link,
                                  "query": pinned or query}
                        if verbose:
                            print(f"  [texture] {category:<10} download -> {asset_id} (CC0)")
            except Exception as e:
                if verbose:
                    print(f"  [texture] {category:<10} download failed ({e.__class__.__name__}), fallback")

        if not out_path.exists():
            _make_procedural(category, query, out_path)
            source = {"provider": "procedural", "license": "generated", "query": query}
            if verbose:
                print(f"  [texture] {category:<10} procedural (offline fallback)")

        manifest[category] = source
        result[category] = out_path

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    # 단독 실행: config.json 을 읽어 모든 텍스처를 미리 받아둔다.
    import argparse
    ap = argparse.ArgumentParser(description="Download/prepare per-category material textures")
    ap.add_argument("--config", "-c", default="./config.json")
    ap.add_argument("--textures", "-t", default="./textures")
    ap.add_argument("--no-download", action="store_true", help="procedural only (offline)")
    ap.add_argument("--distinct", action="store_true",
                    help="클래스 구분용 고채도 텍스처를 textures/distinct 에 생성")
    ap.add_argument("--force", action="store_true", help="distinct 텍스처 강제 재생성")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    categories = [k for k in cfg.keys() if k not in RESERVED_KEYS]
    distinct_colors = {c: cfg[c].get("distinct_color") for c in categories
                       if isinstance(cfg[c], dict) and cfg[c].get("distinct_color")}
    if args.distinct:
        make_distinct_textures(cfg, categories, Path(args.textures) / "distinct",
                               distinct_colors=distinct_colors, force=args.force)
    else:
        ensure_textures(cfg, args.textures, allow_download=not args.no_download,
                        categories=categories)
    print("Done.")
