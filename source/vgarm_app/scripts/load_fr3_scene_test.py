import os
from pathlib import Path

import mujoco


def main() -> None:
    scene_dir = Path("mujoco_menagerie-main") / "mujoco_menagerie-main" / "franka_fr3"
    xml_path = scene_dir / "scene.xml"
    xml = xml_path.read_text(encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        print("loaded_from_string", model.nq, model.nu)
        return
    except Exception as e:
        print("from_xml_string_failed", type(e).__name__, e)

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print("loaded_from_path", model.nq, model.nu)
        return
    except Exception as e:
        print("from_xml_path_failed", type(e).__name__, e)

    os.chdir(str(scene_dir))
    model = mujoco.MjModel.from_xml_string(xml)
    print("loaded_from_string_after_chdir", model.nq, model.nu)


if __name__ == "__main__":
    main()

