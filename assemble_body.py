import mujoco
import mujoco.viewer
import trimesh
from trimesh.transformations import quaternion_matrix, quaternion_from_matrix
from pathlib import Path
import numpy as np


def mirror(spec, meshdir):
    """Mirror a tree of bodies across the Z-plane in-place."""
    mirrored_meshes = set()
    queue = list(spec.worldbody.bodies)

    while queue:
        body = queue.pop(0)
        queue += body.bodies

        body.pos *= [1, 1, -1]

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
                mesh_file = (meshdir / mesh.file).resolve()
                mirrored_file = mesh_file.parent / (
                    mesh_file.stem + "_mirrored" + mesh_file.suffix
                )

                stl = trimesh.load_mesh(mesh_file)
                stl.apply_transform(
                    trimesh.transformations.reflection_matrix([0, 0, 0], [0, 0, 1])
                )
                stl.export(mirrored_file)

                mesh.file = str(mirrored_file.relative_to(meshdir))
                mirrored_meshes.add(geom.meshname)


if __name__ == "__main__":
    model_path = Path("torso/myotorso.xml")
    meshdir = (model_path.parent / "..").resolve()

    spec = mujoco.MjSpec.from_file(str(model_path))

    l_arm_spec = mujoco.MjSpec.from_file("arm/myoarm.xml")
    r_arm_spec = mujoco.MjSpec.from_file("arm/myoarm.xml")
    head = mujoco.MjSpec.from_file("head/myohead.xml")
    legs = mujoco.MjSpec.from_file("leg/myolegs.xml")

    mirror(l_arm_spec, meshdir)

    torso = spec.body("torso")

    l_attach_site = torso.add_site(name="arm_l_attach")
    spec.attach(l_arm_spec, suffix="_l", site=l_attach_site)

    r_attach_site = torso.add_site(name="arm_r_attach")
    spec.attach(r_arm_spec, suffix="_r", site=r_attach_site)

    head_attach_site = torso.add_site(name="head_attach")
    spec.attach(head, site=head_attach_site)

    body = spec.body("Full Body")

    leg_attach_site = body.add_site(name="leg_attach")
    spec.attach(legs, site=leg_attach_site)

    model = spec.compile()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
