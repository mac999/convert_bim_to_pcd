"""
webviewer.py
============
output 폴더의 점군(LAS/LAZ/TXT)을 웹 브라우저에서 탐색/관찰하는 Flask 뷰어.

  - 좌측: output 폴더 트리 네비게이션 (모델 / train / 클래스별 점군)
  - 중앙: Three.js 3D 캔버스 (궤도 카메라, 그리드, 축)
  - 우측: 렌더 설정(점 크기, 투명도, 색상 모드, 최대 점수)
          + 파일 기본정보(점 수, 경계상자, 용량) + 클래스 범례(표시 토글)
  - 다크/라이트 테마 토글 (기본 다크)

사용:
  python webviewer.py -o ./output -c ./config.json -p 5013
  python convert_ifc_to_las.py --viewer-only
"""

import io
import json
import struct
import argparse
from pathlib import Path

import numpy as np
import laspy
from flask import Flask, jsonify, request, render_template, abort, Response

ALLOWED_EXT = {".las", ".laz", ".txt"}
RESERVED_KEYS = {"settings"}
_CACHE = {}          # rel_path -> (mtime, dict(points, colors, classes))
_CACHE_MAX = 3


# ---------------------------------------------------------------------------
# 점군 로딩
# ---------------------------------------------------------------------------
def _load_las(path):
    las = laspy.read(str(path))
    pts = np.vstack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).T
    n = pts.shape[0]
    try:
        cols = np.vstack([np.asarray(las.red), np.asarray(las.green),
                          np.asarray(las.blue)]).T
        cols = (cols // 257).astype(np.uint8)  # 16-bit -> 8-bit
    except Exception:
        cols = np.full((n, 3), 200, np.uint8)
    try:
        cls = np.asarray(las.classification, dtype=np.uint8)
    except Exception:
        cls = np.zeros(n, np.uint8)
    return pts, cols, cls


def _load_txt(path):
    # x,y,z,red,green,blue,classification (헤더 1줄, 콤마 구분)
    try:
        data = np.loadtxt(str(path), delimiter=",", skiprows=1, ndmin=2)
    except Exception:
        data = np.loadtxt(str(path), ndmin=2)  # 공백 구분 fallback
    pts = data[:, 0:3].astype(np.float64)
    n = pts.shape[0]
    cols = (data[:, 3:6].astype(np.int64).clip(0, 255).astype(np.uint8)
            if data.shape[1] >= 6 else np.full((n, 3), 200, np.uint8))
    cls = (data[:, 6].astype(np.int64).clip(0, 255).astype(np.uint8)
           if data.shape[1] >= 7 else np.zeros(n, np.uint8))
    return pts, cols, cls


def load_cloud(abs_path, rel_key):
    mtime = abs_path.stat().st_mtime
    hit = _CACHE.get(rel_key)
    if hit and hit[0] == mtime:
        return hit[1]
    if abs_path.suffix.lower() in (".las", ".laz"):
        pts, cols, cls = _load_las(abs_path)
    else:
        pts, cols, cls = _load_txt(abs_path)
    data = {"points": pts, "colors": cols, "classes": cls}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[rel_key] = (mtime, data)
    return data


# ---------------------------------------------------------------------------
# Flask 앱
# ---------------------------------------------------------------------------
def create_app(output_dir, config_path=None):
    output_dir = Path(output_dir).resolve()
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    def safe_path(rel):
        p = (output_dir / rel).resolve()
        if output_dir not in p.parents and p != output_dir:
            abort(403)
        if not p.exists():
            abort(404)
        return p

    def file_node(f):
        return {"name": f.name,
                "path": f.relative_to(output_dir).as_posix(),
                "type": "cloud",
                "format": f.suffix.lower().lstrip("."),
                "size": f.stat().st_size}

    @app.route("/")
    def index():
        return render_template("viewer.html")

    @app.route("/api/tree")
    def api_tree():
        tree = []
        for model_dir in sorted(output_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            node = {"name": model_dir.name, "type": "model", "children": []}
            for f in sorted(model_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in ALLOWED_EXT:
                    node["children"].append(file_node(f))
            train = model_dir / "train"
            if train.is_dir():
                tnode = {"name": "train", "type": "folder", "children": []}
                labels = train / "labels.json"
                if labels.exists():
                    tnode["labels"] = labels.relative_to(output_dir).as_posix()
                for cat_dir in sorted(train.iterdir()):
                    if not cat_dir.is_dir():
                        continue
                    files = [file_node(f) for f in sorted(cat_dir.iterdir())
                             if f.is_file() and f.suffix.lower() in ALLOWED_EXT]
                    if files:
                        tnode["children"].append(
                            {"name": cat_dir.name, "type": "class", "children": files})
                node["children"].append(tnode)
            if node["children"]:
                tree.append(node)
        return jsonify(tree)

    @app.route("/api/config")
    def api_config():
        cats = []
        lighting = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            lighting = cfg.get("settings", {}).get("lighting", {}) if isinstance(
                cfg.get("settings"), dict) else {}
            names = [k for k in cfg.keys() if k not in RESERVED_KEYS]
            for i, name in enumerate(names, start=1):
                d = cfg[name] if isinstance(cfg[name], dict) else {}
                # distinct 모드면 범례도 구분색을 우선 사용
                color = d.get("distinct_color") or d.get("color", [255, 255, 255])
                classes = d.get("classes", [])
                cats.append({"index": i, "name": name,
                             "color": color,
                             "noise": d.get("noise", 0.0),
                             # Name-keyword-only categories carry no IFC class:
                             # expose the user-defined category name instead,
                             # matching train/labels.json.
                             "ifc_classes": classes or [name],
                             "user_defined": not classes,
                             "name_keywords": d.get("name_keywords", {})})
        return jsonify({"categories": cats, "lighting": lighting})

    @app.route("/api/labels")
    def api_labels():
        p = safe_path(request.args.get("path", ""))
        if p.suffix.lower() != ".json":
            abort(400)
        return Response(p.read_text(encoding="utf-8"), mimetype="application/json")

    @app.route("/api/points")
    def api_points():
        rel = request.args.get("path", "")
        p = safe_path(rel)
        if p.suffix.lower() not in ALLOWED_EXT:
            abort(400)
        max_pts = int(request.args.get("max", 2_000_000))

        data = load_cloud(p, rel)
        pts, cols, cls = data["points"], data["colors"], data["classes"]
        total = pts.shape[0]
        if total == 0:
            abort(422)

        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        uniq, cnt = np.unique(cls, return_counts=True)
        histogram = {int(u): int(c) for u, c in zip(uniq, cnt)}

        if total > max_pts:
            idx = np.sort(np.random.default_rng(0).choice(total, max_pts, replace=False))
            pts, cols, cls = pts[idx], cols[idx], cls[idx]

        offset = (bbox_min + bbox_max) / 2.0  # float32 정밀도 확보용 중심 이동
        pos = (pts - offset).astype(np.float32)

        header = json.dumps({
            "name": p.name, "path": rel, "format": p.suffix.lower().lstrip("."),
            "count": int(pts.shape[0]), "total": int(total),
            "size": p.stat().st_size,
            "offset": offset.tolist(),
            "bbox_min": (bbox_min - offset).tolist(),
            "bbox_max": (bbox_max - offset).tolist(),
            "classes": histogram,
        }).encode("utf-8")
        header += b" " * (-(4 + len(header)) % 4)  # float32 정렬(4바이트 배수)

        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(header)))
        buf.write(header)
        buf.write(pos.tobytes())
        buf.write(np.ascontiguousarray(cols, np.uint8).tobytes())
        buf.write(np.ascontiguousarray(cls, np.uint8).tobytes())
        return Response(buf.getvalue(), mimetype="application/octet-stream")

    return app


def run_viewer(output_dir, config_path=None, port=5013, host="127.0.0.1"):
    app = create_app(output_dir, config_path)
    print(f"\nPoint cloud web viewer: http://{host}:{port}  (output: {output_dir})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Point cloud output web viewer")
    ap.add_argument("--output", "-o", default="./output", help="점군 output 폴더")
    ap.add_argument("--config", "-c", default="./config.json", help="클래스 설정 파일")
    ap.add_argument("--port", "-p", type=int, default=5013)
    a = ap.parse_args()
    run_viewer(a.output, a.config, a.port)
