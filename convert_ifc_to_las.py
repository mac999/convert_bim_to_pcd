"""
convert_ifc_to_las.py
=====================
IFC(BIM) 모델을 파싱/분석하여

  1) IFC product 클래스(카테고리)별로 어울리는 재질 텍스처를 확보하고
     (ambientCG CC0 다운로드 또는 절차적 생성 — texture_manager.py)
  2) 각 객체 서페이스에 삼중평면(triplanar) 투영으로 UV 를 부여해 텍스처를
     자연스럽게 매핑한 **FBX** 를 생성하고 (fbx_exporter.py)
  3) 각 면을 포아송 샘플링해 만든 점군에 텍스처 색상을 입혀
     실 좌표(미터) 스케일 + RGB 를 가진 **LAS / LAZ** 로 저장한다.

LAS classification 필드에는 카테고리 인덱스를 기록해 시맨틱 라벨로 활용할 수 있다.

사용 예:
  python convert_ifc_to_las.py -i ./input -o ./output -c ./config.json --spacing 0.03
"""

import os
import glob
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import laspy
from tqdm import tqdm

import texture_manager
import uv_mapping
from fbx_exporter import FbxSceneBuilder


# ---------------------------------------------------------------------------
# 설정 로딩
# ---------------------------------------------------------------------------
RESERVED_KEYS = {"settings"}


def load_category_mapping(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    settings = config.get("settings", {}) if isinstance(config.get("settings"), dict) else {}
    categories = [k for k in config.keys() if k not in RESERVED_KEYS]

    mapping = {}      # ifc_class -> category
    predef_map = {}   # (ifc_class, PREDEFINED_TYPE) -> category (세부 재분류)
    colors = {}       # category  -> [r,g,b]
    distinct = {}     # category  -> [r,g,b] (구분용 고채도 색)
    uv_scales = {}    # category  -> meters per tile
    transp = {}       # category  -> transparency 0..1
    noise = {}        # category  -> noise level 0..1 (sigma = noise * spacing)
    for category in categories:
        data = config[category]
        if isinstance(data, list):
            ifc_classes, color, uvs, tr, nz = data, [255, 255, 255], 2.0, 0.0, 0.0
            dc = None
            predef = {}
        else:
            ifc_classes = data.get("classes", [])
            color = data.get("color", [255, 255, 255])
            dc = data.get("distinct_color")
            uvs = data.get("uv_scale", 2.0)
            tr = data.get("transparency", 0.0)
            nz = float(data.get("noise", 0.0))
            # predefined: 공유 IFC 클래스를 PredefinedType 으로 세분해 이 카테고리로 재분류
            # 예) roof.predefined = {"IfcSlab": ["ROOF"]}  → 지붕용 슬래브를 roof 로
            predef = data.get("predefined", {}) if isinstance(data.get("predefined"), dict) else {}
        colors[category] = color
        if dc:
            distinct[category] = dc
        uv_scales[category] = uvs
        transp[category] = tr
        noise[category] = min(max(nz, 0.0), 1.0)
        for ifc_class in ifc_classes:
            mapping[ifc_class] = category
        for ifc_class, types in predef.items():
            for t in types:
                predef_map[(ifc_class, str(t).upper())] = category
    return (config, categories, mapping, predef_map, colors, distinct,
            uv_scales, transp, noise, settings)


def resolve_entity_category(entity, ifc_class, mapping, predef_map):
    """PredefinedType 세부 재분류를 우선 적용해 엔티티의 카테고리를 결정."""
    if predef_map:
        pt = getattr(entity, "PredefinedType", None)
        if pt is not None:
            cat = predef_map.get((ifc_class, str(pt).upper()))
            if cat is not None:
                return cat
    return mapping.get(ifc_class)


# ---------------------------------------------------------------------------
# 지오메트리 → 점군(+텍스처 색상)
# ---------------------------------------------------------------------------
def sample_entity(verts, faces, tex, uv_scale, origin, spacing, noise_sigma=0.0):
    """
    한 객체의 삼각형들을 포아송 샘플링하고 텍스처 색상을 입힌다.
    noise_sigma > 0 이면 점 좌표에 가우시안 노이즈(m)를 더해 스캐너 오차를 모사한다.
    반환: (points (N,3), colors (N,3) uint8, normals (N,3) float — 셰이딩용 면 법선)
    """
    tris_idx = faces.reshape(-1, 3)
    tri_v = verts[tris_idx]                       # (K,3,3)
    normals = uv_mapping.triangle_normals(tri_v)  # (K,3)

    A = tri_v[:, 0]
    B = tri_v[:, 1]
    C = tri_v[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    points_per_area = 1.0 / (spacing ** 2)

    pts_all, col_all, nrm_all = [], [], []
    for k in range(len(tris_idx)):
        area = areas[k]
        if area <= 0:
            continue
        n_pts = np.random.poisson(area * points_per_area)
        if n_pts <= 0:
            continue
        r1 = np.random.rand(n_pts, 1)
        r2 = np.random.rand(n_pts, 1)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1.0 - u - v
        p = u * A[k] + v * B[k] + w * C[k]        # (n_pts,3) 월드 좌표

        uv = uv_mapping.triplanar_uv(p, normals[k], uv_scale, origin)
        col = uv_mapping.sample_bilinear(tex, uv) if tex is not None else None

        pts_all.append(p)
        nrm_all.append(np.broadcast_to(normals[k], (n_pts, 3)))
        if col is not None:
            col_all.append(col)

    if not pts_all:
        return (np.empty((0, 3)), np.empty((0, 3), np.uint8), np.empty((0, 3)))
    pts = np.vstack(pts_all)
    nrms = np.vstack(nrm_all)
    cols = np.vstack(col_all) if col_all else np.empty((0, 3), np.uint8)
    if noise_sigma > 0.0 and pts.shape[0] > 0:
        pts = pts + np.random.normal(0.0, noise_sigma, pts.shape)
    return pts, cols, nrms


def entity_uvs(verts, faces, uv_scale, origin):
    """FBX 용: 삼각형-정점별 UV (K,3,2) 계산."""
    tris_idx = faces.reshape(-1, 3)
    tri_v = verts[tris_idx]
    normals = uv_mapping.triangle_normals(tri_v)
    uvs = np.empty((len(tris_idx), 3, 2), dtype=np.float32)
    for k in range(len(tris_idx)):
        uvs[k] = uv_mapping.triplanar_uv(tri_v[k], normals[k], uv_scale, origin)
    return tris_idx, uvs


# ---------------------------------------------------------------------------
# LAS / LAZ 저장 (실 좌표 + RGB + classification)
# ---------------------------------------------------------------------------
def save_point_cloud(points, colors, classes, out_base, formats=("las", "laz")):
    if points.shape[0] == 0:
        print("  (no points to write)")
        return

    mins = points.min(axis=0)
    header = laspy.LasHeader(point_format=3, version="1.2")
    # 실 좌표를 mm 정밀도로 보존
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(mins)

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 8-bit RGB → LAS 16-bit (value * 257)
    c = colors.astype(np.uint16) * 257
    las.red = c[:, 0]
    las.green = c[:, 1]
    las.blue = c[:, 2]
    las.classification = np.clip(classes, 0, 31).astype(np.uint8)

    written = []
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        las.write(path)
        written.append(Path(path).name)
    print(f"  point cloud: {points.shape[0]:,} pts -> {', '.join(written)}")


# ---------------------------------------------------------------------------
# 광원 셰이딩 (텍스처 RGB 에 그림자/명암 부여)
# ---------------------------------------------------------------------------
def resolve_light(light_cfg, bbox_min, bbox_max):
    """
    config 의 lighting 설정을 실제 월드 좌표 광원으로 해석.
    position 은 bbox 정규화 좌표(0=min, 1=max, >1=바깥). 기본은 우측 상단.
    반환: dict(type, world_pos, dir(directional), color(0..1), ambient, intensity, double_sided)
          또는 비활성/무효 시 None.
    """
    if not light_cfg or not light_cfg.get("enabled", True):
        return None
    size = np.asarray(bbox_max, float) - np.asarray(bbox_min, float)
    size[size == 0] = 1.0
    pos_norm = np.asarray(light_cfg.get("position", [1.3, 1.3, 1.6]), float)
    world_pos = np.asarray(bbox_min, float) + pos_norm * size
    center = (np.asarray(bbox_min, float) + np.asarray(bbox_max, float)) / 2.0

    d = world_pos - center
    dn = np.linalg.norm(d)
    direction = d / dn if dn > 0 else np.array([0.0, 0.0, 1.0])

    lc = np.asarray(light_cfg.get("color", [255, 255, 255]), float) / 255.0
    lc = lc / max(lc.max(), 1e-6)   # 최대 채널=1 로 정규화(전체 어두워짐 방지, 색조만 반영)

    # point 광원 거리 감쇠(attenuation): att = 1/(kc + kl*dn + kq*dn²)
    # dn = 광원~점 거리 / range. range 기본은 모델 bbox 대각선(스케일 무관 튜닝).
    att_cfg = light_cfg.get("attenuation", {})
    att_cfg = att_cfg if isinstance(att_cfg, dict) else {}
    diag = float(np.linalg.norm(np.asarray(bbox_max, float) - np.asarray(bbox_min, float)))
    ref_range = float(att_cfg.get("range") or 0.0)
    if ref_range <= 0.0:
        ref_range = diag if diag > 0 else 1.0
    attenuation = {
        "enabled": bool(att_cfg.get("enabled", True)),
        "constant": float(att_cfg.get("constant", 1.0)),
        "linear": float(att_cfg.get("linear", 0.0)),
        "quadratic": float(att_cfg.get("quadratic", 0.0)),
        "range": ref_range,
    }

    return {
        "type": str(light_cfg.get("type", "directional")).lower(),
        "world_pos": world_pos,
        "dir": direction,
        "color": lc,
        "ambient": float(light_cfg.get("ambient", 0.4)),
        "intensity": float(light_cfg.get("intensity", 0.9)),
        "double_sided": bool(light_cfg.get("double_sided", False)),
        "attenuation": attenuation,
    }


def point_attenuation(dist, att):
    """point 광원 거리 감쇠 계수 (N,) 또는 스칼라 1.0. dist: 광원~점 거리(m)."""
    if not att or not att.get("enabled", True):
        return 1.0
    dn = np.asarray(dist, float) / max(att.get("range", 1.0), 1e-9)
    denom = att["constant"] + att["linear"] * dn + att["quadratic"] * dn * dn
    return 1.0 / np.clip(denom, 1e-6, None)


def apply_shading(colors, normals, points, light):
    """Lambert 디퓨즈 셰이딩을 텍스처 색에 적용해 명암/그림자 느낌을 준다."""
    if light is None or colors.shape[0] == 0:
        return colors
    n = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)

    if light["type"] == "point":
        Lvec = light["world_pos"][None, :] - points
        dist = np.linalg.norm(Lvec, axis=1)
        L = Lvec / np.clip(dist[:, None], 1e-9, None)
        ndotl = np.einsum("ij,ij->i", n, L)
        att = point_attenuation(dist, light.get("attenuation"))   # (N,) 거리 감쇠
    else:  # directional / sun (평행광 — 거리 감쇠 없음)
        ndotl = n @ light["dir"]
        att = 1.0

    # 면 winding 이 뒤집힌 경우까지 고르게: double_sided 면 절대값, 아니면 그림자(0 클램프)
    ndotl = np.abs(ndotl) if light["double_sided"] else np.clip(ndotl, 0.0, None)

    # 감쇠는 디퓨즈(광원 기여)에만 적용 — ambient 는 균일 유지
    shade = light["ambient"] + light["intensity"] * ndotl * att  # (N,)
    lit = colors.astype(np.float32) * shade[:, None] * light["color"][None, :]
    return np.clip(lit, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 딥러닝 학습용 train 데이터셋 (클래스별 txt/laz + labels.json)
# ---------------------------------------------------------------------------
def save_train_dataset(model_out_dir, source_name, category_list, cat_index,
                       built, cfg, colors, noise, spacing, light=None):
    """
    train/
      labels.json               클래스 라벨링 기본 정보
      <category>/
        <category>.txt          CSV: x,y,z,red,green,blue,classification
        <category>.laz          해당 클래스 점군만
    built: {category: (points (N,3), colors (N,3) uint8)}  — 셰이딩까지 적용된 최종 색
    """
    train_dir = Path(model_out_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    label_info = {
        "source_ifc": f"{source_name}.ifc",
        "created": datetime.now().isoformat(timespec="seconds"),
        "spacing_m": spacing,
        "noise_model": "gaussian, sigma = noise * spacing (m)",
        "shading": ("off" if light is None else
                    f"{light['type']} light, ambient {light['ambient']}, intensity {light['intensity']}"
                    + ("" if light['type'] != "point" or not light.get('attenuation', {}).get('enabled', True)
                       else f", attenuation kc={light['attenuation']['constant']}"
                            f" kl={light['attenuation']['linear']} kq={light['attenuation']['quadratic']}"
                            f" range={light['attenuation']['range']:.2f}m")),
        "txt_format": "x,y,z,red,green,blue,classification (comma separated, header line)",
        "num_classes": 0,
        "total_points": 0,
        "classes": [],
    }

    for cat in category_list:
        if cat not in built:
            continue
        pts, cols = built[cat]
        if pts.shape[0] == 0:
            continue
        cls = np.full(pts.shape[0], cat_index[cat], np.uint8)

        cat_dir = train_dir / cat
        cat_dir.mkdir(exist_ok=True)

        # txt (CSV)
        txt_path = cat_dir / f"{cat}.txt"
        arr = np.hstack([pts, cols.astype(np.float64), cls[:, None].astype(np.float64)])
        np.savetxt(txt_path, arr, fmt="%.4f,%.4f,%.4f,%d,%d,%d,%d",
                   header="x,y,z,red,green,blue,classification", comments="")

        # laz
        save_point_cloud(pts, cols, cls, str(cat_dir / cat), formats=("laz",))

        cat_cfg = cfg.get(cat, {})
        label_info["classes"].append({
            "index": cat_index[cat],
            "name": cat,
            "ifc_classes": cat_cfg.get("classes", []) if isinstance(cat_cfg, dict) else cat_cfg,
            "color": colors.get(cat, [255, 255, 255]),
            "noise": noise.get(cat, 0.0),
            "num_points": int(pts.shape[0]),
            "files": {
                "txt": f"{cat}/{cat}.txt",
                "laz": f"{cat}/{cat}.laz",
            },
        })
        label_info["total_points"] += int(pts.shape[0])

    label_info["num_classes"] = len(label_info["classes"])
    labels_path = train_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    print(f"  train dataset: {label_info['num_classes']} classes, "
          f"{label_info['total_points']:,} pts -> {train_dir}")


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------
def build_ifc_settings(deflection_tol):
    settings = ifcopenshell.geom.settings()
    for key, val in (("use-world-coords", True),
                     ("weld-vertices", True),
                     ("apply-default-materials", True)):
        try:
            settings.set(key, val)
        except Exception:
            pass
    try:
        settings.set("mesher-linear-deflection", deflection_tol)
    except Exception:
        pass
    return settings


def process_ifc(ifc_file, output_dir, cfg, categories, mapping, colors, uv_scales,
                transp, noise, textures, tex_arrays, settings, spacing, formats,
                light_cfg=None, make_fbx=True, make_train=True, clean=True,
                predef_map=None):
    predef_map = predef_map or {}
    filename = Path(ifc_file).stem
    print(f"\nProcessing: {filename}.ifc")
    model = ifcopenshell.open(ifc_file)

    model_out_dir = Path(output_dir) / filename
    # 기본: 재생성 전에 이전 결과(빠진 클래스 train 폴더·옛 포맷·옛 텍스처 등 stale)를
    # 모두 지워 항상 깨끗한 상태로 만든다. --no-clean 이면 덮어쓰기만.
    if clean and model_out_dir.exists():
        shutil.rmtree(model_out_dir)
        print(f"  cleaned previous output: {model_out_dir}")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    category_list = list(categories)
    cat_index = {c: i + 1 for i, c in enumerate(category_list)}  # classification (1..)

    fbx = FbxSceneBuilder() if make_fbx else None
    if fbx:
        # 출력 FBX 와 함께 배포 가능하도록 텍스처를 모델 폴더로 복사(상대경로 참조)
        tex_subdir = model_out_dir / "textures"
        tex_subdir.mkdir(exist_ok=True)
        for cat in category_list:
            src = textures.get(cat)
            if src and Path(src).exists():
                dst = tex_subdir / Path(src).name
                if not dst.exists():
                    shutil.copyfile(src, dst)
                fbx.set_material(cat, dst, colors[cat], transp.get(cat, 0.0),
                                 stored_name=f"textures/{Path(src).name}")
            else:
                fbx.set_material(cat, None, colors[cat], transp.get(cat, 0.0))

    origin = np.zeros(3)  # triplanar UV 원점(월드) — 타일링은 월드에서 연속
    cat_pts = {c: [] for c in category_list}   # category -> [(N,3), ...]
    cat_col = {c: [] for c in category_list}   # category -> [(N,3) uint8, ...]
    cat_nrm = {c: [] for c in category_list}   # category -> [(N,3), ...] 셰이딩용 법선
    counter = {c: 0 for c in category_list}
    extracted = 0

    # 매핑된 클래스 + 세부 재분류(predefined)에 등장하는 클래스 모두 순회
    supported = list(dict.fromkeys(list(mapping.keys())
                                   + [ic for (ic, _t) in predef_map.keys()]))
    pbar = tqdm(supported, desc="Extracting", leave=True)
    for ifc_class in pbar:
        try:
            entities = model.by_type(ifc_class)
        except RuntimeError:
            # 해당 스키마(예: IFC2X3)에 없는 클래스는 건너뜀
            continue
        if not entities:
            continue
        pbar.set_postfix({"class": ifc_class, "n": len(entities)})

        for entity in entities:
            try:
                # 엔티티별 카테고리 결정(PredefinedType 세부 재분류 우선)
                category = resolve_entity_category(entity, ifc_class, mapping, predef_map)
                if category is None:
                    continue
                tex = tex_arrays.get(category)
                uv_scale = uv_scales.get(category, 2.0)
                noise_sigma = noise.get(category, 0.0) * spacing  # 노이즈 표준편차(m)

                shape = ifcopenshell.geom.create_shape(settings, entity)
                verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, dtype=np.int64)
                if verts.shape[0] == 0 or faces.shape[0] == 0:
                    continue

                # 점군 (텍스처 색상 + 클래스별 노이즈 + 셰이딩용 법선)
                pts, cols, nrms = sample_entity(verts, faces, tex, uv_scale, origin,
                                                spacing, noise_sigma=noise_sigma)
                if pts.shape[0] > 0:
                    cat_pts[category].append(pts)
                    cat_nrm[category].append(nrms)
                    if cols.shape[0] == pts.shape[0]:
                        cat_col[category].append(cols)
                    else:  # 텍스처 없음 → 카테고리 대표색
                        cat_col[category].append(
                            np.tile(np.array(colors[category], np.uint8),
                                    (pts.shape[0], 1)))

                # FBX 메쉬
                if fbx:
                    tris_idx, uvs = entity_uvs(verts, faces, uv_scale, origin)
                    counter[category] += 1
                    name = f"{category}_{counter[category]}"
                    fbx.add_mesh(category, name, verts, tris_idx, uvs)

                extracted += 1
            except Exception:
                continue

    # 카테고리별 배열 결합
    built = {}   # category -> (points, colors, normals)
    for c in category_list:
        if cat_pts[c]:
            built[c] = (np.vstack(cat_pts[c]), np.vstack(cat_col[c]), np.vstack(cat_nrm[c]))

    if built:
        # 전체 bbox 로 광원 위치(정규화) 해석 → 클래스별 셰이딩 적용
        all_min = np.min([v[0].min(axis=0) for v in built.values()], axis=0)
        all_max = np.max([v[0].max(axis=0) for v in built.values()], axis=0)
        light = resolve_light(light_cfg, all_min, all_max)
        if light is not None:
            att = light.get("attenuation", {})
            att_str = ("" if light["type"] != "point" or not att.get("enabled", True)
                       else f", atten kc={att['constant']} kl={att['linear']} "
                            f"kq={att['quadratic']} range={att['range']:.2f}m")
            print(f"  lighting: {light['type']} @ norm {list(light_cfg.get('position', [1.3,1.3,1.6]))}"
                  f" (ambient {light['ambient']}, intensity {light['intensity']}{att_str})")
            for c in built:
                pts, cols, nrms = built[c]
                built[c] = (pts, apply_shading(cols, nrms, pts, light), nrms)

        # 전체 통합 점군
        order = [c for c in category_list if c in built]
        points = np.vstack([built[c][0] for c in order])
        colors_arr = np.vstack([built[c][1] for c in order])
        classes = np.concatenate(
            [np.full(built[c][0].shape[0], cat_index[c], np.uint8) for c in order])
        save_point_cloud(points, colors_arr, classes,
                         str(model_out_dir / filename), formats=formats)

        # 딥러닝 세그먼테이션 학습용 클래스별 데이터셋 (셰이딩 적용된 색)
        if make_train:
            train_built = {c: (built[c][0], built[c][1]) for c in built}
            save_train_dataset(model_out_dir, filename, category_list, cat_index,
                               train_built, cfg, colors, noise, spacing, light=light)
    else:
        print("  (no geometry extracted)")

    if fbx:
        fbx_path = model_out_dir / f"{filename}.fbx"
        try:
            fbx.save(fbx_path, fmt="fbx")
            print(f"  textured mesh: {extracted} objects -> {fbx_path.name}")
        except Exception as e:
            print(f"  FBX save failed: {e}")

    print(f"  total objects extracted: {extracted}")


def main():
    parser = argparse.ArgumentParser(
        description="IFC(BIM) -> textured FBX + RGB LAS/LAZ point cloud converter")
    parser.add_argument("--input", "-i", default="./input", help="IFC 입력 폴더")
    parser.add_argument("--output", "-o", default="./output", help="출력 폴더")
    parser.add_argument("--config", "-c", default="./config.json", help="카테고리/텍스처 설정")
    parser.add_argument("--textures", default="./textures", help="텍스처 캐시 폴더")
    parser.add_argument("--spacing", "-s", type=float, default=0.03,
                        help="점군 샘플 간격(m) (default 0.03)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.005,
                        help="메쉬 선형 편향 허용치(m) (default 0.005)")
    parser.add_argument("--formats", default="las,laz",
                        help="점군 출력 포맷 콤마구분 (las,laz)")
    parser.add_argument("--no-download", action="store_true",
                        help="텍스처 온라인 다운로드 비활성(절차적 생성만)")
    parser.add_argument("--no-fbx", action="store_true", help="FBX 생성 생략")
    parser.add_argument("--no-train", action="store_true",
                        help="클래스별 train 데이터셋 생성 생략")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 난수 시드")
    parser.add_argument("--viewer", action="store_true",
                        help="변환 후 output 폴더 웹 뷰어(Flask) 실행")
    parser.add_argument("--viewer-only", action="store_true",
                        help="변환 없이 웹 뷰어만 실행")
    parser.add_argument("--port", type=int, default=5013, help="웹 뷰어 포트 (default 5013)")
    parser.add_argument("--texture-mode", choices=["realistic", "distinct"], default=None,
                        help="텍스처 방식. distinct=클래스 구분용 고채도 (config settings.texture_mode 덮어씀)")
    parser.add_argument("--no-shading", action="store_true", help="광원 셰이딩 비활성")
    parser.add_argument("--no-clean", action="store_true",
                        help="모델 출력 폴더를 재생성 전에 지우지 않고 덮어쓰기만 (기본은 clean)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.config):
        raise SystemExit(f"config not found: {args.config}")

    if not args.viewer_only:
        (cfg, categories, mapping, predef_map, colors, distinct, uv_scales,
         transp, noise, settings_cfg) = load_category_mapping(args.config)

        tex_mode = args.texture_mode or settings_cfg.get("texture_mode", "realistic")
        light_cfg = None if args.no_shading else settings_cfg.get("lighting")

        print(f"Preparing textures (mode={tex_mode})...")
        textures = texture_manager.ensure_textures(
            cfg, args.textures, allow_download=not args.no_download,
            mode=tex_mode, categories=categories, distinct_colors=distinct)
        tex_arrays = {}
        for cat, path in textures.items():
            try:
                tex_arrays[cat] = uv_mapping.load_texture_array(path)
            except Exception:
                tex_arrays[cat] = None

        settings = build_ifc_settings(args.tolerance)
        formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())

        ifc_files = glob.glob(os.path.join(args.input, "*.ifc")) + \
            glob.glob(os.path.join(args.input, "*.IFC"))
        ifc_files = sorted(set(ifc_files))
        if not ifc_files:
            print(f"No IFC files found in '{args.input}'.")
        else:
            print(f"Settings -> spacing {args.spacing}m | tolerance {args.tolerance}m | "
                  f"formats {formats} | fbx {not args.no_fbx} | train {not args.no_train} | "
                  f"shading {light_cfg is not None and light_cfg.get('enabled', True)} | "
                  f"clean {not args.no_clean}")
            for ifc_file in ifc_files:
                process_ifc(ifc_file, args.output, cfg, categories, mapping, colors,
                            uv_scales, transp, noise, textures, tex_arrays, settings,
                            args.spacing, formats, light_cfg=light_cfg,
                            make_fbx=not args.no_fbx, make_train=not args.no_train,
                            clean=not args.no_clean, predef_map=predef_map)
            print("\nAll done.")

    if args.viewer or args.viewer_only:
        from webviewer import run_viewer
        run_viewer(args.output, args.config, port=args.port)


if __name__ == "__main__":
    main()
