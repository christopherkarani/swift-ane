import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def source_schema_header(name: str) -> str:
    command = (
        f"source {str(SCRIPTS_DIR / 'autoresearch_results_schema.sh')!s} && "
        f"printf '%s' \"${name}\""
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class AutoresearchResultsSchemaTests(unittest.TestCase):
    def test_schema_header_includes_new_compare_fields(self) -> None:
        header = source_schema_header("AUTORESEARCH_RESULTS_TSV_HEADER")
        self.assertIn("espresso_compile_retry_count", header)
        self.assertIn("espresso_compile_failure_count", header)
        self.assertIn("espresso_exact_head_backend", header)
        self.assertIn("espresso_cached_bindings_enabled", header)
        self.assertIn("output_dir", header)
        self.assertIn("prompt_id", header)
        self.assertIn("change_summary", header)

    def test_setup_lane_upgrades_legacy_header_and_rows(self) -> None:
        legacy_header = source_schema_header("AUTORESEARCH_RESULTS_TSV_HEADER_LEGACY_WITH_PROMPT_ID")
        expected_header = source_schema_header("AUTORESEARCH_RESULTS_TSV_HEADER")

        with tempfile.TemporaryDirectory() as directory:
            worktree = Path(directory) / "lane-repo"
            worktree.mkdir()
            subprocess.run(["git", "init", str(worktree)], check=True, capture_output=True, text=True)

            results_path = worktree / "custom-results.tsv"
            results_path.write_text(
                legacy_header
                + "\n"
                + "\t".join(
                    [
                        "2026-03-26T00:00:00Z",
                        "abc1234",
                        "measured",
                        "espresso_tokens_per_second",
                        "10.0",
                        "8.0",
                        "1.25",
                        "true",
                        "true",
                        "2.0",
                        "3.0",
                        "4.0",
                        "5.0",
                        "6.0",
                        "7.0",
                        "8.0",
                        "9.0",
                        "/tmp/out",
                        "prompt-a",
                        "summary",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "bash",
                    str(SCRIPTS_DIR / "setup_autoresearch_lane.sh"),
                    "benchmark-referee",
                    "--worktree",
                    str(worktree),
                    "--local-dir",
                    ".autoresearch-test",
                    "--results-file",
                    results_path.name,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            rows = results_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rows[0], expected_header)
        upgraded = rows[1].split("\t")
        self.assertEqual(len(upgraded), len(expected_header.split("\t")))
        self.assertEqual(upgraded[17:21], ["", "", "", ""])
        self.assertEqual(upgraded[21], "/tmp/out")
        self.assertEqual(upgraded[22], "prompt-a")
        self.assertEqual(upgraded[23], "summary")


if __name__ == "__main__":
    unittest.main()
