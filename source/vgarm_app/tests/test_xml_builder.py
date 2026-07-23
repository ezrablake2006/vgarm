import unittest

from vgarm.mjc import available_robots, build_scene_xml
from vgarm.reconstruction.types import SceneLayout, SimObject


class TestXMLBuilder(unittest.TestCase):
    def test_build_has_equality_constraints(self):
        robots = available_robots()
        robot = robots["franka_fr3"]
        scene = SceneLayout(
            objects=[
                SimObject(
                    name="cube_red",
                    category="cube",
                    geom_type="box",
                    pos_xyz=(0.55, 0.0, 0.03),
                    size_xyz=(0.02, 0.02, 0.02),
                    rgba=(0.9, 0.2, 0.2, 1.0),
                )
            ]
        )
        built = build_scene_xml(scene, robot)
        self.assertIn('connect name="attach_cube_red"', built.xml_text)
        self.assertIn('site1="attachment_site"', built.xml_text)
        self.assertIn('site2="cube_red_grab"', built.xml_text)

    def test_kinova_uses_existing_terminal_pinch_site_and_top_down_orientation(self):
        robot = available_robots()["kinova_gen3"]
        self.assertEqual(robot.attachment_site_name, "pinch_site")
        self.assertIn("kinova_gen3", str(robot.include_xml_path))
        self.assertEqual(robot.grasp_orientation[-3:], (0.0, 0.0, -1.0))


if __name__ == "__main__":
    unittest.main()
