import os
import tempfile
from supabase import create_client
import rhino3dm
import pyvista as pv

# Initialize Supabase client
def _init_supabase():
    url = os.getenv('SUPABASE_URL', 'https://apdbfbjnlsxjfubqahtl.supabase.co')
    key = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzQ3NjUzOCwiZXhwIjoyMDYzMDUyNTM4fQ.cylQZjLEmtBi507wrJ1KUDyIXTz5H5VAXGj5eKEqDy4')
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be set')
    return create_client(url, key)

supabase = _init_supabase()
BUCKET = os.getenv('SUPABASE_BUCKET', 'models')


def convert_to_glb(file_key: str) -> str:
    """
    Downloads a .3dm file from Supabase storage, converts it to a GLB, and returns the local file path.
    """
    # Download model bytes
    table = supabase.storage.from_(BUCKET)
    result = table.download(file_key)
    if result.error:
        raise RuntimeError(f"Error downloading {file_key}: {result.error.message}")
    data = result.data

    # Write to temp .3dm
    tmp_3dm = tempfile.NamedTemporaryFile(delete=False, suffix='.3dm')
    tmp_3dm.write(data)
    tmp_3dm.flush()
    tmp_3dm.close()

    # Read model
    model = rhino3dm.File3dm.Read(tmp_3dm.name)

    # Collect meshes
    meshes = []
    for obj in model.Objects:
        geom = obj.Geometry
        if isinstance(geom, rhino3dm.Brep):
            # mesh all faces
            for face in geom.Faces:
                try:
                    mesh = face.GetMesh(rhino3dm.MeshType.Any)
                    if mesh:
                        # convert to PyVista PolyData
                        pts = [(v.X, v.Y, v.Z) for v in mesh.Vertices]
                        faces = []
                        for f in mesh.Faces:
                            if len(f) == 4:
                                idxs = (f[0], f[1], f[2], f[3])
                            else:
                                idxs = (f[0], f[1], f[2])
                            faces.append((len(idxs),) + idxs)
                        faces_flat = [i for face in faces for i in face]
                        pv_mesh = pv.PolyData(pts, faces_flat)
                        meshes.append(pv_mesh)
                except Exception:
                    continue

    if not meshes:
        raise RuntimeError('No geometry found for conversion')

    # Combine meshes into one
    multiblock = pv.MultiBlock(meshes)
    combined = multiblock.combine()

    # Export to GLB
    tmp_glb = tempfile.NamedTemporaryFile(delete=False, suffix='.glb')
    glb_path = tmp_glb.name
    tmp_glb.close()
    pv.save_meshio(glb_path, combined)

    return glb_path
