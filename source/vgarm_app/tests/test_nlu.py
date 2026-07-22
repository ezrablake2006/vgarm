import unittest

from vgarm.nlu import parse_cn


class TestNLU(unittest.TestCase):
    def test_parse_pick_place(self):
        it = parse_cn("把红色方块移到左边")
        self.assertEqual(it.kind, "move_to_dir")
        self.assertEqual(it.object_color, "red")
        self.assertEqual(it.object_category, "cube")
        self.assertEqual(it.target_keyword, "左边")


if __name__ == "__main__":
    unittest.main()
