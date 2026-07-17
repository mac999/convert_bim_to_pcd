"""
convert_ifc_to_las.py
=====================
Converts an IFC (BIM) model into:

  1) per-category material textures (ambientCG CC0 download or procedural - texture_manager.py)
  2) a textured FBX with triplanar-projected UVs (fbx_exporter.py)
  3) a Poisson-sampled RGB point cloud in real-world meters, saved as LAS/LAZ

The LAS classification field holds the category index, usable as a semantic label.

Example:
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
import shading_fx
from fbx_exporter import FbxSceneBuilder


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
RESERVED_KEYS = {"settings"}


def load_category_mapping(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    settings = config.get("settings", {}) if isinstance(config.get("settings"), dict) else {}
    categories = [k for k in config.keys() if k not in RESERVED_KEYS]

    mapping = {}      # ifc_class -> category
    predef_map = {}   # (ifc_class, PREDEFINED_TYPE) -> category
    colors = {}       # category  -> [r,g,b]
    distinct = {}     # category  -> [r,g,b] (high-saturation, for distinct mode)
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
            # Re-route a shared IFC class into this category by PredefinedType.
            # e.g. roof.predefined = {"IfcSlab": ["ROOF"]} sends roof slabs to roof.
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
    """Resolve an entity's category, letting PredefinedType override the class mapping."""
    if predef_map:
        pt = getattr(entity, "PredefinedType", None)
        if pt is not None:
            cat = predef_map.get((ifc_class, str(pt).upper()))
            if cat is not None:
                return cat
    return mapping.get(ifc_class)


# ---------------------------------------------------------------------------
# Geometry -> point cloud (+ texture color)
# ---------------------------------------------------------------------------
def sample_entity(verts, faces, tex, uv_scale, origin, spacing, noise_sigma=0.0,
                  fx=None, dmap=None, fx_params=None):
    """
    Poisson-sample one object's triangles and tint the points with the texture.
    noise_sigma > 0 adds Gaussian position noise (m) to mimic scanner error.

    With fx (a shading_fx.load_fx_config result), also applies the cheap
    photorealism effects: normal-map perturbation, cavity darkening, and edge AO.

    Returns (points (N,3), colors (N,3) uint8, normals (N,3), ao (N,)).
    """
    tris_idx = faces.reshape(-1, 3)
    tri_v = verts[tris_idx]                       # (K,3,3)
    normals = uv_mapping.triangle_normals(tri_v)  # (K,3)

    A = tri_v[:, 0]
    B = tri_v[:, 1]
    C = tri_v[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    points_per_area = 1.0 / (spacing ** 2)

    nrm_strength, detail_amt, ao_strength, convex_hl = fx_params or (0.0, 0.0, 0.0, 0.0)
    use_bump = bool(fx and fx["normal_map"]["enabled"] and dmap is not None)
    use_ao = bool(fx and fx["edge_ao"]["enabled"] and ao_strength > 0.0)

    crease = height = None
    if use_ao:
        crease, height = shading_fx.build_creases(verts, tris_idx, normals, tri_v, fx["edge_ao"])
        use_ao = crease is not None
    ao_radius = fx["edge_ao"]["radius"] if fx else 0.1

    pts_all, col_all, nrm_all, ao_all = [], [], [], []
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
        p = u * A[k] + v * B[k] + w * C[k]        # (n_pts,3) world coords

        uv = uv_mapping.triplanar_uv(p, normals[k], uv_scale, origin)
        col = uv_mapping.sample_bilinear(tex, uv) if tex is not None else None

        if use_bump:
            n_pt = shading_fx.perturb_normal(normals[k], uv, dmap, nrm_strength)
        else:
            n_pt = np.broadcast_to(normals[k], (n_pts, 3))

        # Cavity shadow multiplies the sampled color directly, leaving downstream untouched
        if col is not None and use_bump and detail_amt > 0.0:
            d = shading_fx.detail_shadow_factor(uv, dmap, detail_amt)
            col = np.clip(col.astype(np.float32) * d[:, None], 0, 255).astype(np.uint8)

        if use_ao:
            bary = np.hstack([u, v, w])
            ao_all.append(shading_fx.edge_ao_factor(bary, crease[k], height[k],
                                                    ao_radius, ao_strength, convex_hl))
        else:
            ao_all.append(np.ones(n_pts, np.float32))

        pts_all.append(p)
        nrm_all.append(n_pt)
        if col is not None:
            col_all.append(col)

    if not pts_all:
        return (np.empty((0, 3)), np.empty((0, 3), np.uint8),
                np.empty((0, 3)), np.empty(0, np.float32))
    pts = np.vstack(pts_all)
    nrms = np.vstack(nrm_all)
    ao = np.concatenate(ao_all)
    cols = np.vstack(col_all) if col_all else np.empty((0, 3), np.uint8)
    if noise_sigma > 0.0 and pts.shape[0] > 0:
        pts = pts + np.random.normal(0.0, noise_sigma, pts.shape)
    return pts, cols, nrms, ao


def entity_uvs(verts, faces, uv_scale, origin):
    """Per triangle-vertex UVs (K,3,2) for the FBX exporter."""
    tris_idx = faces.reshape(-1, 3)
    tri_v = verts[tris_idx]
    normals = uv_mapping.triangle_normals(tri_v)
    uvs = np.empty((len(tris_idx), 3, 2), dtype=np.float32)
    for k in range(len(tris_idx)):
        uvs[k] = uv_mapping.triplanar_uv(tri_v[k], normals[k], uv_scale, origin)
    return tris_idx, uvs


# ---------------------------------------------------------------------------
# LAS / LAZ writing (real coords + RGB + classification)
# ---------------------------------------------------------------------------
def save_point_cloud(points, colors, classes, out_base, formats=("las", "laz")):
    if points.shape[0] == 0:
        print("  (no points to write)")
        return

    mins = points.min(axis=0)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])   # keep mm precision
    header.offsets = np.floor(mins)

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 8-bit RGB -> LAS 16-bit (value * 257)
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
# Light shading (adds shading/contrast to the texture RGB)
# ---------------------------------------------------------------------------
def resolve_light(light_cfg, bbox_min, bbox_max):
    """
    Turn the config lighting block into a world-space light.
    position is bbox-normalized (0=min, 1=max, >1=outside).
    Returns a light dict, or None when disabled.
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
    lc = lc / max(lc.max(), 1e-6)   # max channel = 1: keep hue, don't darken

    # Point-light attenuation: att = 1/(kc + kl*dn + kq*dn^2), dn = distance / range.
    # range defaults to the model bbox diagonal so tuning is scale-independent.
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
    """Point-light distance attenuation (N,), or scalar 1.0 when disabled."""
    if not att or not att.get("enabled", True):
        return 1.0
    dn = np.asarray(dist, float) / max(att.get("range", 1.0), 1e-9)
    denom = att["constant"] + att["linear"] * dn + att["quadratic"] * dn * dn
    return 1.0 / np.clip(denom, 1e-6, None)


def apply_shading(colors, normals, points, light, ao=None, fx=None, space_lights=None):
    """
    Apply Lambert diffuse shading to the texture colors.

    With fx, ambient becomes a hemisphere lerp, IfcSpace interior lights are added,
    and edge AO occludes indirect light fully / direct light by direct_weight.
    With fx=None the result is identical to plain Lambert (uniform ambient * light color).
    """
    if light is None or colors.shape[0] == 0:
        return colors
    n = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)

    if light["type"] == "point":
        Lvec = light["world_pos"][None, :] - points
        dist = np.linalg.norm(Lvec, axis=1)
        L = Lvec / np.clip(dist[:, None], 1e-9, None)
        ndotl = np.einsum("ij,ij->i", n, L)
        att = point_attenuation(dist, light.get("attenuation"))
    else:  # directional / sun: parallel rays, no distance falloff
        ndotl = n @ light["dir"]
        att = 1.0

    # double_sided uses |n.L| so flipped winding still lights up; otherwise clamp to 0
    ndotl = np.abs(ndotl) if light["double_sided"] else np.clip(ndotl, 0.0, None)

    direct = (light["intensity"] * ndotl * att)[:, None] * light["color"][None, :]   # (N,3)
    # Without fx, hemisphere_ambient() returns the light color -> same as the old formula
    ambient = light["ambient"] * shading_fx.hemisphere_ambient(n, fx, light["color"])

    interior = (shading_fx.interior_contribution(points, n, space_lights, fx,
                                                 double_sided=light["double_sided"])
                if space_lights else 0.0)

    if ao is not None and fx and fx["edge_ao"]["enabled"]:
        a = ao[:, None]
        dw = fx["edge_ao"]["direct_weight"]
        ambient = ambient * a
        interior = interior * a
        direct = direct * (1.0 - dw * (1.0 - a))

    lit = colors.astype(np.float32) * (ambient + direct + interior)
    return np.clip(lit, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training dataset (per-class txt/laz + labels.json)
# ---------------------------------------------------------------------------
def save_train_dataset(model_out_dir, source_name, category_list, cat_index,
                       built, cfg, colors, noise, spacing, light=None, fx=None):
    """
    train/
      labels.json               class label metadata
      <category>/<category>.txt CSV: x,y,z,red,green,blue,classification
      <category>/<category>.laz points of that class only
    built: {category: (points (N,3), colors (N,3) uint8)} with shading already applied.
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
        "shading_fx": shading_fx.describe(fx),
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

        txt_path = cat_dir / f"{cat}.txt"
        arr = np.hstack([pts, cols.astype(np.float64), cls[:, None].astype(np.float64)])
        np.savetxt(txt_path, arr, fmt="%.4f,%.4f,%.4f,%d,%d,%d,%d",
                   header="x,y,z,red,green,blue,classification", comments="")

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
# Main pipeline
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
                predef_map=None, fx=None, detail_maps=None):
    predef_map = predef_map or {}
    detail_maps = detail_maps or {}
    filename = Path(ifc_file).stem
    print(f"\nProcessing: {filename}.ifc")
    model = ifcopenshell.open(ifc_file)

    model_out_dir = Path(output_dir) / filename
    # Wipe stale output (dropped class folders, old formats/textures) unless --no-clean
    if clean and model_out_dir.exists():
        shutil.rmtree(model_out_dir)
        print(f"  cleaned previous output: {model_out_dir}")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    category_list = list(categories)
    cat_index = {c: i + 1 for i, c in enumerate(category_list)}  # classification (1..)

    fbx = FbxSceneBuilder() if make_fbx else None
    if fbx:
        # Copy textures next to the FBX so it stays portable (relative references)
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

    space_lights = shading_fx.collect_space_lights(model, settings, fx) if fx else []

    origin = np.zeros(3)  # triplanar UV origin: tiling stays continuous in world space
    cat_pts = {c: [] for c in category_list}
    cat_col = {c: [] for c in category_list}
    cat_nrm = {c: [] for c in category_list}
    cat_ao = {c: [] for c in category_list}
    counter = {c: 0 for c in category_list}
    extracted = 0

    # Walk mapped classes plus any class referenced by a predefined rule
    supported = list(dict.fromkeys(list(mapping.keys())
                                   + [ic for (ic, _t) in predef_map.keys()]))
    pbar = tqdm(supported, desc="Extracting", leave=True)
    for ifc_class in pbar:
        try:
            entities = model.by_type(ifc_class)
        except RuntimeError:
            continue          # class absent from this schema (e.g. IFC2X3)
        if not entities:
            continue
        pbar.set_postfix({"class": ifc_class, "n": len(entities)})

        for entity in entities:
            try:
                category = resolve_entity_category(entity, ifc_class, mapping, predef_map)
                if category is None:
                    continue
                tex = tex_arrays.get(category)
                uv_scale = uv_scales.get(category, 2.0)
                noise_sigma = noise.get(category, 0.0) * spacing

                shape = ifcopenshell.geom.create_shape(settings, entity)
                verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, dtype=np.int64)
                if verts.shape[0] == 0 or faces.shape[0] == 0:
                    continue

                fx_params = (shading_fx.category_fx(fx, cfg.get(category))
                             if fx else None)
                pts, cols, nrms, ao = sample_entity(
                    verts, faces, tex, uv_scale, origin, spacing,
                    noise_sigma=noise_sigma, fx=fx,
                    dmap=detail_maps.get(category), fx_params=fx_params)
                if pts.shape[0] > 0:
                    cat_pts[category].append(pts)
                    cat_nrm[category].append(nrms)
                    cat_ao[category].append(ao)
                    if cols.shape[0] == pts.shape[0]:
                        cat_col[category].append(cols)
                    else:  # no texture -> fall back to the category color
                        cat_col[category].append(
                            np.tile(np.array(colors[category], np.uint8),
                                    (pts.shape[0], 1)))

                if fbx:
                    tris_idx, uvs = entity_uvs(verts, faces, uv_scale, origin)
                    counter[category] += 1
                    name = f"{category}_{counter[category]}"
                    fbx.add_mesh(category, name, verts, tris_idx, uvs)

                extracted += 1
            except Exception:
                continue

    built = {}   # category -> (points, colors, normals, ao)
    for c in category_list:
        if cat_pts[c]:
            built[c] = (np.vstack(cat_pts[c]), np.vstack(cat_col[c]),
                        np.vstack(cat_nrm[c]), np.concatenate(cat_ao[c]))

    if built:
        # Resolve the normalized light position against the full model bbox
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
            if fx:
                print(f"  shading fx: {shading_fx.describe(fx)}")
            for c in built:
                pts, cols, nrms, ao = built[c]
                built[c] = (pts, apply_shading(cols, nrms, pts, light, ao=ao, fx=fx,
                                               space_lights=space_lights), nrms, ao)

        order = [c for c in category_list if c in built]
        points = np.vstack([built[c][0] for c in order])
        colors_arr = np.vstack([built[c][1] for c in order])
        classes = np.concatenate(
            [np.full(built[c][0].shape[0], cat_index[c], np.uint8) for c in order])
        save_point_cloud(points, colors_arr, classes,
                         str(model_out_dir / filename), formats=formats)

        if make_train:
            train_built = {c: (built[c][0], built[c][1]) for c in built}
            save_train_dataset(model_out_dir, filename, category_list, cat_index,
                               train_built, cfg, colors, noise, spacing, light=light, fx=fx)
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
    parser.add_argument("--input", "-i", default="./input", help="IFC input folder")
    parser.add_argument("--output", "-o", default="./output", help="output folder")
    parser.add_argument("--config", "-c", default="./config.json",
                        help="category/texture configuration")
    parser.add_argument("--textures", default="./textures", help="texture cache folder")
    parser.add_argument("--spacing", "-s", type=float, default=0.03,
                        help="point cloud sample spacing in m (default 0.03)")
    parser.add_argument("--tolerance", "-t", type=float, default=0.005,
                        help="mesh linear deflection tolerance in m (default 0.005)")
    parser.add_argument("--formats", default="las,laz",
                        help="point cloud output formats, comma separated (las,laz)")
    parser.add_argument("--no-download", action="store_true",
                        help="disable online texture download (procedural only)")
    parser.add_argument("--no-fbx", action="store_true", help="skip FBX generation")
    parser.add_argument("--no-train", action="store_true",
                        help="skip the per-class train dataset")
    parser.add_argument("--seed", type=int, default=42, help="sampling random seed")
    parser.add_argument("--viewer", action="store_true",
                        help="launch the Flask web viewer after conversion")
    parser.add_argument("--viewer-only", action="store_true",
                        help="launch the web viewer without converting")
    parser.add_argument("--port", type=int, default=5013, help="web viewer port (default 5013)")
    parser.add_argument("--texture-mode", choices=["realistic", "distinct"], default=None,
                        help="texture style. distinct = high-saturation per class "
                             "(overrides config settings.texture_mode)")
    parser.add_argument("--no-shading", action="store_true", help="disable light shading")
    parser.add_argument("--no-fx", action="store_true",
                        help="disable the cheap photorealism effects (normal map / edge AO / "
                             "hemisphere ambient / interior lights). "
                             "Overrides config settings.shading_fx")
    parser.add_argument("--no-clean", action="store_true",
                        help="overwrite the model output folder instead of wiping it first")
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

        # Preprocess the normal/cavity maps once per category
        fx = None if args.no_fx else shading_fx.load_fx_config(settings_cfg)
        detail_maps = {}
        if fx and fx["normal_map"]["enabled"]:
            for cat, arr in tex_arrays.items():
                detail_maps[cat] = shading_fx.build_detail_map(
                    arr, cavity_scale=fx["normal_map"]["cavity_scale"])
            print(f"  [fx] detail maps built for {len(detail_maps)} categories")

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
                  f"clean {not args.no_clean}\n"
                  f"Shading fx -> {shading_fx.describe(fx)}")
            for ifc_file in ifc_files:
                process_ifc(ifc_file, args.output, cfg, categories, mapping, colors,
                            uv_scales, transp, noise, textures, tex_arrays, settings,
                            args.spacing, formats, light_cfg=light_cfg,
                            make_fbx=not args.no_fbx, make_train=not args.no_train,
                            clean=not args.no_clean, predef_map=predef_map,
                            fx=fx, detail_maps=detail_maps)
            print("\nAll done.")

    if args.viewer or args.viewer_only:
        from webviewer import run_viewer
        run_viewer(args.output, args.config, port=args.port)


if __name__ == "__main__":
    main()
