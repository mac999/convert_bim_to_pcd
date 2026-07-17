"""
shading_fx.py
=============
Cheap real-time-game techniques that add photorealistic cues to the point cloud
colors. Everything reuses data the pipeline already has (texture RGB, face
normals, barycentric coordinates) - no ray tracing or SSAO.

  1) Normal map (bump) + cavity
     Texture luminance is treated as a heightmap; its gradient perturbs the face
     normal, so mortar joints and wood grain catch light. The high-frequency
     component (height minus a blurred copy) darkens recesses. Preprocessed once
     per texture; sampling costs one extra bilinear fetch.

  2) Crease-based edge AO
     Shared mesh edges are classified concave (valley) / convex (ridge) by
     dihedral angle, then shaded by distance to the edge. The distance is nearly
     free because the sampler already has barycentric coordinates:
       d(point, edge opposite vertex i) = bary[i] * (2*area / |opposite edge|)

  3) Hemisphere ambient
     Ambient becomes a sky/ground color lerp driven by n.z instead of a flat
     constant - one lerp.

  4) Interior point lights (IfcSpace)
     A point light is placed near the ceiling of each IfcSpace bbox and applied
     only to points inside that space.

All parameters live in config.json under settings.shading_fx.
"""

import numpy as np
from PIL import Image

import uv_mapping


# ---------------------------------------------------------------------------
# Config loading / normalization
# ---------------------------------------------------------------------------
def _norm_color(rgb, fallback=(255, 255, 255)):
    """[0..255] RGB -> (3,) float normalized so the max channel is 1 (hue only)."""
    c = np.asarray(rgb if rgb else fallback, float) / 255.0
    return c / max(c.max(), 1e-6)


def load_fx_config(settings):
    """Parse settings.shading_fx into a normalized dict, or None when disabled."""
    cfg = settings.get("shading_fx") if isinstance(settings, dict) else None
    if not isinstance(cfg, dict) or not cfg.get("enabled", True):
        return None

    def sub(name):
        d = cfg.get(name)
        return d if isinstance(d, dict) else {}

    nm, ao, hemi, itl = sub("normal_map"), sub("edge_ao"), sub("hemisphere_ambient"), sub("interior_light")
    itl_att = itl.get("attenuation") if isinstance(itl.get("attenuation"), dict) else {}

    return {
        "normal_map": {
            "enabled": bool(nm.get("enabled", True)),
            "strength": float(nm.get("strength", 1.0)),
            "detail_shadow": float(nm.get("detail_shadow", 0.5)),
            "cavity_scale": int(nm.get("cavity_scale", 12)),
        },
        "edge_ao": {
            "enabled": bool(ao.get("enabled", True)),
            "angle_threshold_deg": float(ao.get("angle_threshold_deg", 35.0)),
            "smooth_deg": float(ao.get("smooth_deg", 25.0)),
            "radius": float(ao.get("radius", 0.12)),
            "strength": float(ao.get("strength", 0.55)),
            "convex_highlight": float(ao.get("convex_highlight", 0.12)),
            "boundary_as_crease": bool(ao.get("boundary_as_crease", True)),
            "direct_weight": float(ao.get("direct_weight", 0.5)),
        },
        "hemisphere_ambient": {
            "enabled": bool(hemi.get("enabled", True)),
            "sky": _norm_color(hemi.get("sky_color"), (150, 175, 210)),
            "ground": _norm_color(hemi.get("ground_color"), (95, 85, 75)),
            "intensity": float(hemi.get("intensity", 1.0)),
        },
        "interior_light": {
            "enabled": bool(itl.get("enabled", True)),
            "source": str(itl.get("source", "ifcspace")).lower(),
            "color": _norm_color(itl.get("color"), (255, 236, 200)),
            "intensity": float(itl.get("intensity", 0.55)),
            "height_ratio": float(itl.get("height_ratio", 0.85)),
            "margin": float(itl.get("margin", 0.3)),
            "max_lights": int(itl.get("max_lights", 64)),
            "min_volume": float(itl.get("min_volume", 1.0)),
            "range_ratio": float(itl.get("range_ratio", 0.5)),
            "attenuation": {
                "constant": float(itl_att.get("constant", 1.0)),
                "linear": float(itl_att.get("linear", 0.7)),
                "quadratic": float(itl_att.get("quadratic", 1.8)),
            },
        },
    }


def category_fx(fx, cat_cfg):
    """
    Apply the per-category normal_strength / ao_strength overrides.
    Returns (normal_strength, detail_shadow, ao_strength, convex_highlight).
    """
    if not isinstance(cat_cfg, dict):
        cat_cfg = {}
    ns = float(cat_cfg.get("normal_strength", 1.0))
    aos = float(cat_cfg.get("ao_strength", 1.0))
    nm, ao = fx["normal_map"], fx["edge_ao"]
    return (nm["strength"] * ns, nm["detail_shadow"] * ns,
            ao["strength"] * aos, ao["convex_highlight"] * aos)


# ---------------------------------------------------------------------------
# 1) Normal map + cavity (preprocessed once per texture)
# ---------------------------------------------------------------------------
def build_detail_map(tex, cavity_scale=12):
    """
    Texture (H,W,3) uint8 -> (H,W,3) float32 detail map:
      ch0 = dh/du, ch1 = dh/dv, ch2 = cavity (-1..1, negative = recessed)
    Height h is texture luminance. Cavity is h minus a blurred copy, so joints
    and seams come out negative.
    """
    if tex is None:
        return None
    t = np.asarray(tex, np.float32) / 255.0
    h = 0.299 * t[:, :, 0] + 0.587 * t[:, :, 1] + 0.114 * t[:, :, 2]

    # UV v runs opposite to image rows, hence the sign flip
    g_row, g_col = np.gradient(h)
    dh_du = g_col
    dh_dv = -g_row

    # Downsample + upsample = cheap box blur, to isolate the high-frequency part
    hh, ww = h.shape
    s = max(int(cavity_scale), 2)
    small = Image.fromarray((np.clip(h, 0, 1) * 255).astype(np.uint8)).resize(
        (max(ww // s, 1), max(hh // s, 1)), Image.BILINEAR)
    blur = np.asarray(small.resize((ww, hh), Image.BILINEAR), np.float32) / 255.0
    cavity = np.clip((h - blur) * 4.0, -1.0, 1.0)

    return np.stack([dh_du, dh_dv, cavity], axis=2).astype(np.float32)


def _tangent_frame(normal):
    """
    (T, B) tangent frame matching the triplanar axis choice in uv_mapping._axis_uv:
      axis 0 -> u=Y, v=Z / axis 1 -> u=X, v=Z / axis 2 -> u=X, v=Y
    Gram-Schmidt orthogonalized against the normal.
    """
    axis = int(np.argmax(np.abs(normal)))
    if axis == 0:
        t, b = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    elif axis == 1:
        t, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    else:
        t, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = normal / max(np.linalg.norm(normal), 1e-9)
    t = t - n * float(n @ t)
    tn = np.linalg.norm(t)
    if tn < 1e-6:
        t = b - n * float(n @ b)
        tn = max(np.linalg.norm(t), 1e-9)
    t = t / tn
    b = np.cross(n, t)
    return t, b


def perturb_normal(normal, uv, dmap, strength, gain=6.0):
    """Bump the face normal: n' = normalize(n - s*(dh/du * T + dh/dv * B)). Returns (N,3)."""
    n = normal / max(np.linalg.norm(normal), 1e-9)
    if dmap is None or strength <= 0.0:
        return np.broadcast_to(n, (uv.shape[0], 3))
    d = uv_mapping.sample_bilinear_f(dmap, uv)     # (N,3) = (dh_du, dh_dv, cavity)
    t, b = _tangent_frame(n)
    grad = (d[:, 0:1] * t[None, :] + d[:, 1:2] * b[None, :]) * (strength * gain)
    pn = n[None, :] - grad
    return pn / np.clip(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9, None)


def detail_shadow_factor(uv, dmap, amount):
    """Cavity darkening multiplier (N,); 1.0 when there is no map."""
    if dmap is None or amount <= 0.0:
        return 1.0
    cav = uv_mapping.sample_bilinear_f(dmap, uv)[:, 2]
    return np.clip(1.0 + amount * np.minimum(cav, 0.0), 0.05, 1.0)


# ---------------------------------------------------------------------------
# 2) Crease-based edge AO (preprocessed once per object)
# ---------------------------------------------------------------------------
def build_creases(verts, tris_idx, tri_normals, tri_v, ao_cfg):
    """
    Find crease edges. Returns (crease (K,3), height (K,3)) or (None, None).
      crease[k,i] : strength of the edge opposite vertex i of triangle k.
                    >0 concave (AO), <0 convex (highlight), 0 flat. |v| = angle weight.
      height[k,i] : 2*area / |that edge|. Times the barycentric weight = point-edge distance.
    """
    K = len(tris_idx)
    if K == 0:
        return None, None

    thresh = np.radians(max(ao_cfg["angle_threshold_deg"], 0.0))
    smooth = np.radians(max(ao_cfg["smooth_deg"], 1.0))

    # Edge i is the one opposite vertex i
    edges = np.concatenate([tris_idx[:, [1, 2]], tris_idx[:, [2, 0]], tris_idx[:, [0, 1]]], axis=0)
    tri_id = np.tile(np.arange(K), 3)
    loc_id = np.repeat(np.arange(3), K)

    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    key = lo.astype(np.int64) * (len(verts) + 1) + hi.astype(np.int64)

    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))

    crease = np.zeros((K, 3), np.float32)

    # Shared edges: dihedral angle decides crease, neighbor's far vertex decides the sign
    pair = cnt == 2
    if np.any(pair):
        i0 = order[starts[pair]]
        i1 = order[starts[pair] + 1]
        t0, t1 = tri_id[i0], tri_id[i1]
        l0, l1 = loc_id[i0], loc_id[i1]
        n0, n1 = tri_normals[t0], tri_normals[t1]
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", n0, n1), -1.0, 1.0))
        w = np.clip((ang - thresh) / smooth, 0.0, 1.0)

        # The neighbor's vertex not on the shared edge is tris_idx[t1, l1]
        far1 = verts[tris_idx[t1, l1]]
        cent0 = tri_v[t0].mean(axis=1)
        # Neighbor in front of the plane -> valley (concave), behind -> ridge (convex)
        sign = np.where(np.einsum("ij,ij->i", n0, far1 - cent0) > 0, 1.0, -1.0)
        val = (sign * w).astype(np.float32)
        crease[t0, l0] = val
        crease[t1, l1] = val

    # Open boundary edges have no neighbor; optionally treat them as concave
    if ao_cfg["boundary_as_crease"]:
        solo = cnt == 1
        if np.any(solo):
            i0 = order[starts[solo]]
            crease[tri_id[i0], loc_id[i0]] = 1.0

    if not np.any(crease):
        return None, None

    A, B, C = tri_v[:, 0], tri_v[:, 1], tri_v[:, 2]
    cross = np.cross(B - A, C - A)
    area2 = np.linalg.norm(cross, axis=1)             # = 2 * area
    elen = np.stack([np.linalg.norm(C - B, axis=1),
                     np.linalg.norm(A - C, axis=1),
                     np.linalg.norm(B - A, axis=1)], axis=1)
    height = (area2[:, None] / np.clip(elen, 1e-9, None)).astype(np.float32)
    return crease, height


def edge_ao_factor(bary, crease_k, height_k, radius, strength, convex_highlight):
    """
    Edge AO multiplier (N,) for points inside one triangle.
      bary : (N,3) barycentric weights [u,v,w]
      crease_k / height_k : row k of the build_creases output
    Distance to an edge is bary[i] * height[i], attenuated by exp(-d / radius).
    """
    occ = None
    hl = None
    inv_r = 1.0 / max(radius, 1e-6)
    for i in range(3):
        c = float(crease_k[i])
        if c == 0.0:
            continue
        f = np.exp(-(bary[:, i] * height_k[i]) * inv_r) * abs(c)
        if c > 0:
            occ = f if occ is None else np.maximum(occ, f)
        else:
            hl = f if hl is None else np.maximum(hl, f)

    factor = np.ones(bary.shape[0], np.float32)
    if occ is not None:
        factor = factor - strength * occ
    if hl is not None:
        factor = factor + convex_highlight * hl
    return np.clip(factor, 0.0, 2.0)


# ---------------------------------------------------------------------------
# 3) Hemisphere ambient
# ---------------------------------------------------------------------------
def hemisphere_ambient(normals, fx, fallback_color):
    """
    (N,3) ambient color lerped sky/ground by n.z.
    When disabled, returns the light color so callers keep their previous behavior.
    """
    hemi = fx["hemisphere_ambient"] if fx else None
    if not hemi or not hemi["enabled"]:
        return np.broadcast_to(np.asarray(fallback_color, np.float32), (normals.shape[0], 3))
    t = (0.5 + 0.5 * normals[:, 2])[:, None]          # n.z: -1..1 -> 0..1
    col = hemi["ground"][None, :] * (1.0 - t) + hemi["sky"][None, :] * t
    return (col * hemi["intensity"]).astype(np.float32)


# ---------------------------------------------------------------------------
# 4) Interior point lights (IfcSpace)
# ---------------------------------------------------------------------------
def collect_space_lights(model, geom_settings, fx, verbose=True):
    """
    Place one ceiling light per IfcSpace bbox.
    Returns [{pos, bmin, bmax, range, volume}, ...].
    """
    itl = fx["interior_light"] if fx else None
    if not itl or not itl["enabled"] or itl["source"] != "ifcspace":
        return []
    import ifcopenshell.geom

    try:
        spaces = model.by_type("IfcSpace")
    except RuntimeError:
        return []

    lights = []
    for sp in spaces:
        try:
            shape = ifcopenshell.geom.create_shape(geom_settings, sp)
            v = np.array(shape.geometry.verts, np.float64).reshape(-1, 3)
            if v.shape[0] == 0:
                continue
            mn, mx = v.min(axis=0), v.max(axis=0)
            size = mx - mn
            vol = float(np.prod(np.clip(size, 1e-6, None)))
            if vol < itl["min_volume"]:
                continue
            pos = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0,
                            mn[2] + itl["height_ratio"] * size[2]])
            m = itl["margin"]
            diag = float(np.linalg.norm(size))
            lights.append({
                "pos": pos,
                "bmin": mn - m,
                "bmax": mx + m,
                "range": max(diag * itl["range_ratio"], 0.5),
                "volume": vol,
            })
        except Exception:
            continue

    lights.sort(key=lambda d: d["volume"], reverse=True)
    dropped = max(len(lights) - itl["max_lights"], 0)
    lights = lights[:itl["max_lights"]]
    if verbose and lights:
        print(f"  interior lights: {len(lights)} from IfcSpace"
              + (f" ({dropped} smaller spaces dropped by max_lights)" if dropped else ""))
    return lights


def interior_contribution(points, normals, lights, fx, double_sided=False):
    """
    Diffuse contribution (N,3) of the interior lights. Each light only affects points
    inside its own space bbox - a miniature of tiled/clustered lighting.
    """
    out = np.zeros((points.shape[0], 3), np.float32)
    if not lights or not fx:
        return out
    itl = fx["interior_light"]
    att = itl["attenuation"]
    for L in lights:
        m = np.all((points >= L["bmin"]) & (points <= L["bmax"]), axis=1)
        if not np.any(m):
            continue
        P, N = points[m], normals[m]
        vec = L["pos"][None, :] - P
        dist = np.linalg.norm(vec, axis=1)
        ndotl = np.einsum("ij,ij->i", N, vec / np.clip(dist[:, None], 1e-9, None))
        ndotl = np.abs(ndotl) if double_sided else np.clip(ndotl, 0.0, None)
        dn = dist / L["range"]
        a = 1.0 / np.clip(att["constant"] + att["linear"] * dn + att["quadratic"] * dn * dn, 1e-6, None)
        out[m] += (itl["intensity"] * ndotl * a)[:, None] * itl["color"][None, :]
    return out


def describe(fx):
    """One-line summary for the console log and labels.json."""
    if not fx:
        return "off"
    parts = []
    if fx["normal_map"]["enabled"]:
        parts.append(f"normal_map(strength {fx['normal_map']['strength']}, "
                     f"detail_shadow {fx['normal_map']['detail_shadow']})")
    if fx["edge_ao"]["enabled"]:
        parts.append(f"edge_ao(angle {fx['edge_ao']['angle_threshold_deg']}deg, "
                     f"radius {fx['edge_ao']['radius']}m, strength {fx['edge_ao']['strength']})")
    if fx["hemisphere_ambient"]["enabled"]:
        parts.append("hemisphere_ambient")
    if fx["interior_light"]["enabled"]:
        parts.append(f"interior_light({fx['interior_light']['source']}, "
                     f"intensity {fx['interior_light']['intensity']})")
    return ", ".join(parts) if parts else "off"
