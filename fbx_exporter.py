"""
fbx_exporter.py
===============
aspose.threed 로 텍스처가 매핑된 FBX 씬을 점진적으로 구성/저장하는 헬퍼.

  - 카테고리별로 PhongMaterial(디퓨즈 텍스처) 을 1개씩 생성해 재사용
  - IFC 객체 하나를 Mesh(node) 로 추가하며, 삼각형별 UV 를
    POLYGON_VERTEX/DIRECT 매핑으로 부여
  - 최종적으로 FBX(기본 7400 binary) 로 저장

FBX 는 텍스처 이미지를 파일 경로로 참조하므로, 결과 .fbx 와 textures/ 폴더를
함께 배포해야 뷰어에서 텍스처가 보인다.
"""

from pathlib import Path

from aspose.threed import Scene, FileFormat
from aspose.threed.entities import Mesh, MappingMode, ReferenceMode, TextureMapping
from aspose.threed.utilities import Vector3, Vector4, FVector4
from aspose.threed.shading import PhongMaterial, Texture

_FORMATS = {
    "fbx": FileFormat.FBX7400_BINARY,
    "fbx_ascii": FileFormat.FBX7400ASCII,
    "obj": FileFormat.WAVEFRONT_OBJ,
    "glb": FileFormat.GLTF2_BINARY,
    "gltf": FileFormat.GLTF2,
}


class FbxSceneBuilder:
    def __init__(self):
        self.scene = Scene()
        self._materials = {}
        self._category_parent = {}

    def set_material(self, category, texture_path, color=(200, 200, 200),
                     transparency=0.0, stored_name=None):
        """
        카테고리 재질 등록(1회). 디퓨즈 텍스처 + 베이스 컬러/투명도.
          texture_path : 존재 확인용 실제 경로
          stored_name  : FBX 에 기록할 텍스처 경로(보통 FBX 기준 상대경로).
                         None 이면 절대경로를 기록.
        """
        mat = PhongMaterial()
        mat.diffuse_color = Vector3(1.0, 1.0, 1.0)  # 텍스처 원색을 그대로 표현
        mat.specular_color = Vector3(0.1, 0.1, 0.1)
        mat.shininess = 12.0
        if transparency and transparency > 0:
            mat.transparency = float(transparency)
        tex_path = Path(texture_path) if texture_path else None
        if tex_path is not None and tex_path.exists():
            tex = Texture()
            tex.file_name = stored_name if stored_name else str(tex_path.resolve())
            tex.name = category
            mat.set_texture(PhongMaterial.MAP_DIFFUSE, tex)
        self._materials[category] = mat

    def _parent_node(self, category):
        if category not in self._category_parent:
            self._category_parent[category] = self.scene.root_node.create_child_node(category)
        return self._category_parent[category]

    def add_mesh(self, category, name, verts, tris, uvs):
        """
        객체 하나를 씬에 추가.
          verts : (M,3) float  월드 좌표(m)
          tris  : (K,3) int    삼각형 정점 인덱스
          uvs   : (K,3,2) float 삼각형-정점별 UV
        """
        if len(tris) == 0:
            return
        mesh = Mesh()
        cp = mesh.control_points
        for v in verts:
            cp.append(Vector4(float(v[0]), float(v[1]), float(v[2]), 1.0))

        uv_elem = mesh.create_element_uv(TextureMapping.DIFFUSE)
        uv_elem.mapping_mode = MappingMode.POLYGON_VERTEX
        uv_elem.reference_mode = ReferenceMode.DIRECT
        uv_data = uv_elem.data

        for k, tri in enumerate(tris):
            mesh.create_polygon([int(tri[0]), int(tri[1]), int(tri[2])])
            for j in range(3):
                u, w = uvs[k, j]
                uv_data.append(FVector4(float(u), float(w), 0.0, 0.0))

        node = self._parent_node(category).create_child_node(name, mesh)
        if category in self._materials:
            node.material = self._materials[category]

    def save(self, path, fmt="fbx"):
        fileformat = _FORMATS.get(fmt, FileFormat.FBX7400_BINARY)
        self.scene.save(str(path), fileformat)
