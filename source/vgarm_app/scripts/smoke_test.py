import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco

from vgarm.mjc import PickPlaceExecutor, available_robots, build_scene_xml
from vgarm.nlu import parse_cn
from vgarm.reconstruction import reconstruct_scene


def main() -> None:
    layout = reconstruct_scene(scene_json_path="vgarm_app/examples/scenes/sample_scene.json")
    robot = available_robots()["franka_fr3"]
    robot_dir = Path(robot.include_xml_path).resolve().parent
    built = build_scene_xml(layout, robot, xml_base_dir=robot_dir)
    xml_path = robot_dir / f"_vgarm_scene_{robot.robot_id}.xml"
    xml_path.write_text(built.xml_text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    cmd = parse_cn("把红色方块移到左边")
    execu = PickPlaceExecutor(model, data, robot, built.object_names)
    obj = [o for o in layout.objects if "red" in o.name][0]
    xy = execu.resolve_target_xy(cmd.target_keyword)
    execu.pick_and_place(obj.name, xy)
    for _ in range(50):
        mujoco.mj_step(model, data)
    print("smoke_ok", model.nq, model.nv, model.nu)


if __name__ == "__main__":
    main()
