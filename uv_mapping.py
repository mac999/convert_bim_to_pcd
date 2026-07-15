"""
uv_mapping.py
=============
IFC 지오메트리는 일반적으로 UV 좌표를 포함하지 않으므로,
월드 좌표에 정렬된 삼중평면(triplanar/box) 투영으로 UV 를 생성한다.

핵심 아이디어
  - 각 삼각형의 법선(normal)에서 지배 축(dominant axis)을 고른다.
  - 지배 축에 수직인 평면으로 정점을 투영한 뒤 ``uv_scale``(타일 1장이 덮는 미터)
    로 나누어 UV 를 만든다 → 실제 치수에 비례해 텍스처가 자연스럽게 반복된다.
  - 점군 색상도 동일한 투영으로 각 점의 UV 를 구해 텍스처에서 RGB 를 이중선형 샘플링한다.

이 방식은 벽/바닥/천장처럼 축에 정렬된 면이 많은 BIM 모델에서
이음매 없이(seamless) 타일링되어 시각적으로 자연스럽다.
"""

import numpy as np
from PIL import Image


def triangle_normals(tri_verts):
    """(N,3,3) 삼각형 정점 → (N,3) 단위 법선."""
    a, b, c = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def _axis_uv(points, axis):
    """지배 축(axis: 0=x,1=y,2=z) 기준으로 (N,3) 점을 2D 평면 좌표로 투영."""
    if axis == 0:      # 법선이 X → YZ 평면
        return np.stack([points[:, 1], points[:, 2]], axis=1)
    elif axis == 1:    # 법선이 Y → XZ 평면
        return np.stack([points[:, 0], points[:, 2]], axis=1)
    else:              # 법선이 Z → XY 평면
        return np.stack([points[:, 0], points[:, 1]], axis=1)


def triplanar_uv(points, normal, uv_scale, origin):
    """
    한 삼각형 평면 위 점들의 UV 계산.
      points : (N,3) 월드 좌표
      normal : (3,) 삼각형 단위 법선
      uv_scale : 텍스처 타일 1장이 덮는 실제 거리(m)
      origin : (3,) 투영 기준점(모델 최소 좌표) — UV 원점을 안정화해 이음매 정렬
    반환 : (N,2) UV (0~ 범위, 텍스처 wrap 로 반복)
    """
    axis = int(np.argmax(np.abs(normal)))
    proj = _axis_uv(points - origin, axis)
    uv = proj / max(uv_scale, 1e-6)
    return uv


def load_texture_array(path):
    """텍스처 파일 → (H,W,3) uint8 numpy 배열."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def sample_bilinear(tex, uv):
    """
    텍스처에서 UV(반복/wrap) 위치의 색을 이중선형 샘플링.
      tex : (H,W,3) uint8
      uv  : (N,2) float (임의 범위, wrap 됨)
    반환 : (N,3) uint8
    """
    h, w = tex.shape[:2]
    # UV → 텍셀 좌표 (v 는 상하 반전: 이미지 원점이 좌상단)
    fx = (uv[:, 0] % 1.0) * (w - 1)
    fy = (1.0 - (uv[:, 1] % 1.0)) * (h - 1)

    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (fx - x0)[:, None]
    dy = (fy - y0)[:, None]

    c00 = tex[y0, x0].astype(np.float32)
    c10 = tex[y0, x1].astype(np.float32)
    c01 = tex[y1, x0].astype(np.float32)
    c11 = tex[y1, x1].astype(np.float32)

    top = c00 * (1 - dx) + c10 * dx
    bot = c01 * (1 - dx) + c11 * dx
    col = top * (1 - dy) + bot * dy
    return np.clip(col, 0, 255).astype(np.uint8)
