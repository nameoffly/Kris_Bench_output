import unittest

import translate_ins_only as t


class TranslateInsOnlyTests(unittest.TestCase):
    def test_chunk_list_preserves_order_and_remainder(self):
        self.assertEqual(t.chunk_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_parse_json_array_supports_fenced_json(self):
        text = "```json\n[\"ko text\", \"yo text\"]\n```"
        self.assertEqual(t.parse_json_array(text), ["ko text", "yo text"])

    def test_build_ins_output_updates_only_ins_en(self):
        source = {
            "1": {
                "ori_img": "1.jpg",
                "ins_en": "Turn sky red",
                "explain_en": "Color changes",
            }
        }
        out = t.build_ins_output(source, {"Turn sky red": "하늘을 빨갛게 바꿔라"})
        self.assertEqual(out["1"]["ins_en"], "하늘을 빨갛게 바꿔라")
        self.assertEqual(out["1"]["explain_en"], "Color changes")
        self.assertEqual(out["1"]["ori_img"], "1.jpg")


if __name__ == "__main__":
    unittest.main()
