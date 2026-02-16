import mujoco
import mujoco.viewer
import time
import trimesh
from pathlib import Path
import numpy as np
import tempfile
import os

MIRROR_MAT = np.diag([1.0, 1.0, -1.0])

def mirror_quat(quat):
    """Mirror a quaternion's rotation across the Z-plane: M @ R @ M."""
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    R = mat.reshape(3, 3)
    R_mirrored = MIRROR_MAT @ R @ MIRROR_MAT
    q_out = np.empty(4)
    mujoco.mju_mat2Quat(q_out, R_mirrored.flatten())
    return q_out


def mirror_body_tree(spec, root_bodies, meshdir):
    """Mirror a tree of bodies across the Z-plane in-place."""
    mirrored_meshes = set()
    queue = list(root_bodies)

    while queue:
        body = queue.pop(0)
        queue += body.bodies
        print(body.name)

        body.pos *= [1, 1, -1]
        body.quat[:] = mirror_quat(body.quat)

        for joint in body.joints:
            joint.axis *= [1, 1, -1]
            joint.range = [-joint.range[1], -joint.range[0]]

        for site in body.sites:
            site.pos *= [1, 1, -1]

        for geom in body.geoms:
            geom.pos *= [1, 1, -1]

            if geom.meshname and geom.meshname not in mirrored_meshes:
                mesh = next(m for m in spec.meshes if m.name == geom.meshname)
                mesh_file = (meshdir / mesh.file).resolve()
                mirrored_file = mesh_file.parent / (mesh_file.stem + "_mirrored" + mesh_file.suffix)

                stl = trimesh.load_mesh(mesh_file)
                stl.apply_transform(trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1]))
                stl.export(mirrored_file)

                mesh.file = str(mirrored_file.relative_to(meshdir))
                mirrored_meshes.add(geom.meshname)


if __name__ == "__main__":
    model_path = Path("body/myobody.xml")
    meshdir = (model_path.parent / "..").resolve()

    spec = mujoco.MjSpec.from_file(str(model_path))

    # Load arm as a separate spec for attaching a mirrored copy
    arm_xml = '''<mujoco>
      <compiler angle="radian" meshdir=".." texturedir=".."/>
      <include file="../arm/assets/myoarm_assets.xml"/>
      <worldbody>
        <include file="../arm/assets/myoarm_body.xml"/>
      </worldbody>
    </mujoco>'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=str(model_path.parent), delete=False) as f:
        f.write(arm_xml)
        tmp_path = f.name

    try:
        arm_spec = mujoco.MjSpec.from_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Attach mirrored arm copy to torso with '_l' suffix
    torso = spec.body("torso")
    attach_site = torso.add_site(name="arm_l_attach")
    spec.attach(arm_spec, suffix="_l", site=attach_site)

    # Mirror only the attached left arm bodies
    left_roots = [spec.body("/thorax_l"), spec.body("/clavicle_l")]
    mirror_body_tree(spec, left_roots, meshdir)

    model = spec.compile()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
