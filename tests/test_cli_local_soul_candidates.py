from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import apps.cli.cli as cli


class TestLocalSoulCandidates(unittest.TestCase):
    def test_newest_non_default_first_when_no_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = d / "older.sou"
            b = d / "newer.sou"
            a.write_text("x", encoding="utf-8")
            b.write_text("y", encoding="utf-8")
            os.utime(a, (10, 10))
            os.utime(b, (99_999, 99_999))
            self.assertEqual(cli._local_soul_candidate_paths(d), [b, a])

    def test_canonical_first_then_newest_others(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            a = d / "older.sou"
            b = d / "newer.sou"
            default = d / "sloughgpt.sou"
            a.write_text("x", encoding="utf-8")
            b.write_text("y", encoding="utf-8")
            default.write_text("z", encoding="utf-8")
            os.utime(a, (10, 10))
            os.utime(b, (99_999, 99_999))
            os.utime(default, (50_000, 50_000))
            paths = cli._local_soul_candidate_paths(d)
            self.assertEqual(paths[0], default)
            self.assertEqual(paths[1], b)
            self.assertEqual(paths[2], a)

    def test_missing_dir(self) -> None:
        d = Path(tempfile.mkdtemp()) / "nope"
        self.assertEqual(cli._local_soul_candidate_paths(d), [])


if __name__ == "__main__":
    unittest.main()
