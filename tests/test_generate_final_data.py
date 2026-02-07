import json
import tempfile
import unittest
from pathlib import Path

import generate_final_data as g


class GenerateFinalDataTests(unittest.TestCase):
    def test_build_final_records_adds_id_and_type_with_sorted_ids(self):
        annotation = {
            "10": {"ori_img": "10.jpg", "ins_en": "ten"},
            "2": {"ori_img": "2.jpg", "ins_en": "two"},
        }
        records = g.build_final_records_for_category("mathematics", annotation)
        self.assertEqual([r["id"] for r in records], ["2", "10"])
        self.assertEqual(records[0]["type"], "mathematics")
        self.assertEqual(records[0]["ori_img"], "2.jpg")

    def test_generate_final_data_for_language_merges_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "KRIS_Bench"
            (root / "b_cat").mkdir(parents=True)
            (root / "a_cat").mkdir(parents=True)
            with (root / "a_cat" / "annotation_ko_ins.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"1": {"ori_img": "a1.jpg", "ins_en": "a"}},
                    f,
                    ensure_ascii=False,
                )
            with (root / "b_cat" / "annotation_ko_ins.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {"1": {"ori_img": "b1.jpg", "ins_en": "b"}},
                    f,
                    ensure_ascii=False,
                )
            records = g.generate_final_data_for_language(root, "ko", "ins", strict=True)
            self.assertEqual([r["type"] for r in records], ["a_cat", "b_cat"])
            self.assertEqual(len(records), 2)

    def test_missing_language_file_raises_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "KRIS_Bench"
            (root / "only_cat").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                g.generate_final_data_for_language(root, "yo", "ins", strict=True)


if __name__ == "__main__":
    unittest.main()
