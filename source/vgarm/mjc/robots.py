from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    include_xml_path: Path
    attachment_site_name: str
    default_home_qpos: tuple[float, ...] | None = None
    grasp_orientation: tuple[float, ...] | None = None
    orientation_alignment_steps: int = 0
    transit_max_steps: int | None = None


def _menagerie_root() -> Path:
    configured = os.environ.get("VGARM_MENAGERIE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root / "mujoco_menagerie-main" / "mujoco_menagerie-main"


def available_robots() -> dict[str, RobotSpec]:
    menagerie = _menagerie_root()
    return {
        "franka_fr3": RobotSpec(
            robot_id="franka_fr3",
            include_xml_path=menagerie / "franka_fr3" / "fr3.xml",
            attachment_site_name="attachment_site",
        ),
        "franka_panda": RobotSpec(
            robot_id="franka_panda",
            include_xml_path=menagerie / "franka_emika_panda" / "panda_nohand.xml",
            attachment_site_name="attachment_site",
        ),
        "ur5e": RobotSpec(
            robot_id="ur5e",
            include_xml_path=menagerie / "universal_robots_ur5e" / "ur5e.xml",
            attachment_site_name="attachment_site",
        ),
        "kinova_gen3": RobotSpec(
            robot_id="kinova_gen3",
            include_xml_path=menagerie / "kinova_gen3" / "gen3.xml",
            attachment_site_name="pinch_site",
            # Menagerie's home pose points the pinch axis horizontally.  Rotate
            # the tool to a top-down pose before Cartesian task execution.
            grasp_orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            orientation_alignment_steps=3000,
            transit_max_steps=1400,
        ),
    }
