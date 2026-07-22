import unittest

from vgarm.nlu import parse_cn


class TestNLUIntents(unittest.TestCase):
    def test_move_to_dir_synonyms(self):
        for t in ["把红色方块移到左边", "将红色方块放到左侧", "把红色方块搬到前方"]:
            it = parse_cn(t)
            self.assertIn(it.kind, ("move_to_dir",))
            self.assertIn(it.target_keyword, ("左边", "前面"))

    def test_lift(self):
        for t in ["把红色方块抬起", "将蓝色方块提起", "把黄色方块拿起"]:
            it = parse_cn(t)
            self.assertEqual(it.kind, "lift")

    def test_move_next_to_object(self):
        it = parse_cn("把红色方块放在蓝色方块的左边")
        self.assertEqual(it.kind, "move_next_to_object")
        self.assertEqual(it.target_keyword, "左边")
        self.assertEqual(it.ref_color, "blue")

    def test_swap(self):
        it = parse_cn("交换红色方块和蓝色方块位置")
        self.assertEqual(it.kind, "swap")
        self.assertEqual(it.object_color, "red")
        self.assertEqual(it.ref_color, "blue")


if __name__ == "__main__":
    unittest.main()

