import mujoco
import mujoco.viewer
import trimesh
from trimesh.transformations import quaternion_matrix, quaternion_from_matrix
from pathlib import Path
import numpy as np


def mirror(spec):
    """Mirror a tree of bodies across the Z-plane in-place."""
    mirrored_meshes = set()
    queue = list(spec.worldbody.bodies)

    while queue:
        body = queue.pop(0)
        queue += body.bodies

        body.pos *= [1, 1, -1]

        if body.alt.type == mujoco.mjtOrientation.mjORIENTATION_EULER:
            # For xyz eulerseq: M @ Rx(ex)Ry(ey)Rz(ez) @ M = Rx(-ex)Ry(-ey)Rz(ez)
            body.alt.euler *= [-1, -1, 1]
        else:
            T = np.diag([1.0, 1.0, -1.0, 1.0])
            R = quaternion_matrix(body.quat)
            body.quat[:] = quaternion_from_matrix(T @ R @ T)

        for joint in body.joints:
            joint.axis *= [1, 1, -1]
            joint.range = [-joint.range[1], -joint.range[0]]

        for site in body.sites:
            site.pos *= [1, 1, -1]

        for geom in body.geoms:
            geom.pos *= [1, 1, -1]

            if geom.meshname and geom.meshname not in mirrored_meshes:
                mesh = next(m for m in spec.meshes if m.name == geom.meshname)
                mesh_file = (Path(__file__).parent / mesh.file).resolve()
                mirrored_file = mesh_file.parent / (
                    mesh_file.stem + "_mirrored" + mesh_file.suffix
                )

                stl = trimesh.load_mesh(mesh_file)
                stl.apply_transform(
                    trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])
                )
                stl.export(mirrored_file)

                mesh.file = str(mirrored_file.relative_to(Path(__file__).parent))
                mirrored_meshes.add(geom.meshname)

    return spec


if __name__ == "__main__":
    full_body_spec = mujoco.MjSpec.from_file("base.xml")

    torso_spec = mujoco.MjSpec.from_file("torso/myotorso.xml")
    arm_spec = mujoco.MjSpec.from_file("arm/myoarm.xml")
    left_arm_spec = mirror(mujoco.MjSpec.from_file("arm/myoarm.xml"))
    head_spec = mujoco.MjSpec.from_file("head/myohead.xml")
    legs_spec = mujoco.MjSpec.from_file("leg/myolegs.xml")

    root_body = full_body_spec.body("root")

    root_attach_site = root_body.add_site(name="torso_attach")
    full_body_spec.attach(torso_spec, suffix="_torso", site=root_attach_site)
    full_body_spec.attach(legs_spec, suffix="_leg", site=root_attach_site)

    torso_body = full_body_spec.body("/torso_torso")

    torso_attach_site = torso_body.add_site(name="arm_l_attach")
    full_body_spec.attach(left_arm_spec, suffix="_arm_l", site=torso_attach_site)
    full_body_spec.attach(arm_spec, suffix="_arm_r", site=torso_attach_site)
    full_body_spec.attach(head_spec, suffix="_head", site=torso_attach_site)

    model = full_body_spec.compile()
    data = mujoco.MjData(model)

    full_body_spec.to_file("frank.xml")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
