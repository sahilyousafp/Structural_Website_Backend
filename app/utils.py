import rhino3dm

def mesh_brep(brep, mesh_type=rhino3dm.MeshType.Any):
    """Extract meshes from a Brep."""
    meshes = []
    for face in brep.Faces:
        try:
            m = face.GetMesh(mesh_type)
            if m:
                meshes.append(m)
        except Exception:
            continue
    return meshes
