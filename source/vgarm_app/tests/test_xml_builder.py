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


if __name__ == "__main__":
    unittest.main()

