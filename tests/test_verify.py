"""Regression checks for the compact verifier's failure modes (stdlib only)."""

import csv
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compact_verify", REPOSITORY / "verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data.txt").write_text("evidence\n")
        self.digest = hashlib.sha256((self.root / "data.txt").read_bytes()).hexdigest()

    def manifest(self, text):
        (self.root / "SHA256SUMS.txt").write_text(text)

    def test_valid_relative_manifest(self):
        self.manifest(f"{self.digest}  ./data.txt\n")
        self.assertEqual(verify.verify_manifest(self.root), {"data.txt"})

    def test_corruption_or_missing_file_fails(self):
        for name, digest in (("absent", self.digest), ("data.txt", "0" * 64)):
            with self.subTest(name=name):
                self.manifest(f"{digest}  {name}\n")
                with self.assertRaises(verify.VerificationError):
                    verify.verify_manifest(self.root)

    def test_unsafe_and_duplicate_paths_fail(self):
        for name in ("../data.txt", "/data.txt", "a/../data.txt", "C:/data.txt", "a\\data.txt"):
            with self.subTest(name=name):
                self.manifest(f"{self.digest}  {name}\n")
                with self.assertRaises(verify.VerificationError):
                    verify.verify_manifest(self.root)
        self.manifest(f"{self.digest}  data.txt\n{self.digest}  ./data.txt\n")
        with self.assertRaisesRegex(verify.VerificationError, "Duplicate"):
            verify.verify_manifest(self.root)

    def test_empty_malformed_and_self_referential_manifests_fail(self):
        for text in ("", "not a checksum\n", f"{self.digest}  SHA256SUMS.txt\n"):
            with self.subTest(text=text):
                self.manifest(text)
                with self.assertRaises(verify.VerificationError):
                    verify.verify_manifest(self.root)

    def test_symlink_is_rejected(self):
        (self.root / "alias.txt").symlink_to(self.root / "data.txt")
        self.manifest(f"{self.digest}  alias.txt\n")
        with self.assertRaisesRegex(verify.VerificationError, "Symlink"):
            verify.verify_manifest(self.root)

    def test_markdown_source_policy(self):
        for name in ("paper.tex", "sources.bib", "format.sty", "format.cls", "paper.pdf", "manuscript-v1.pdf"):
            with self.subTest(name=name):
                (self.root / name).write_text("evidence\n")
                self.manifest(f"{self.digest}  {name}\n")
                with self.assertRaisesRegex(verify.VerificationError, "Disallowed"):
                    verify.verify_manifest(self.root)
        # Figure PDFs and archived JSON provenance are data, not manuscript sources.
        for name in ("figure.pdf", "historical.json"):
            (self.root / name).write_text("evidence\n")
            self.manifest(f"{self.digest}  {name}\n")
            self.assertEqual(verify.verify_manifest(self.root), {name})

    def test_optimized_python_still_rejects_corruption(self):
        self.manifest(f"{'0' * 64}  data.txt\n")
        for optimization in ("-O", "-OO"):
            result = subprocess.run([sys.executable, optimization, str(REPOSITORY / "verify.py"),
                                     "--root", str(self.root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("SHA256 mismatch", result.stderr)


class ArithmeticTests(unittest.TestCase):
    def test_exact_test_and_exact_ties(self):
        self.assertEqual(verify.exact_sign_flip([-1] * 8), 2 / 256)
        self.assertEqual(verify.exact_sign_flip([0, 0, -1, -2]), 0.5)
        self.assertEqual(verify.exact_sign_flip([0, 0]), 1)
        # Equal task weighting differs from treating three runs as exchangeable tasks.
        self.assertEqual(verify.exact_sign_flip([2, -3, -3], ["a", "b", "b"]), 1)
        self.assertEqual(verify.exact_sign_flip([2, -3, -3]), 0.5)

    def test_nonfinite_values_and_duplicate_rows_fail(self):
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(verify.VerificationError):
                verify.exact_sign_flip([value])
        with self.assertRaisesRegex(verify.VerificationError, "Duplicate"):
            verify.index_rows([{"id": "one"}, {"id": "one"}], ("id",), "test row")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def copy_family(self, family):
        target = self.root / "evidence" / family / "aggregate"
        shutil.copytree(REPOSITORY / "evidence" / family / "aggregate", target)
        return target

    def change_csv(self, path, mutate):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames, rows = reader.fieldnames, list(reader)
        mutate(rows)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_released_primary_arithmetic(self):
        self.assertIn("480 layers", verify.verify_stage4(REPOSITORY))
        self.assertIn("n=8", verify.verify_real_lora(REPOSITORY))
        self.assertIn("n=10", verify.verify_transformer(REPOSITORY))

    def test_changed_stage4_layer_fails_without_hash_check(self):
        folder = self.copy_family("stage4")
        self.change_csv(folder / "stage4_all_layer_summaries.csv", lambda rows: rows[0].update(gradient_effective_rank="100"))
        with self.assertRaisesRegex(verify.VerificationError, "Stage4 hard_knee mean_r2"):
            verify.verify_stage4(self.root)

    def test_changed_identity_initial_loss_fails_without_hash_check(self):
        folder = self.copy_family("real_lora")
        def mutate(rows):
            row = next(row for row in rows if row["strategy"] == "uniform_identity_control")
            row["initial_val_loss"] = str(float(row["initial_val_loss"]) + 1)
            row["val_loss_delta"] = str(float(row["final_val_loss"]) - float(row["initial_val_loss"]))
        self.change_csv(folder / "all_results.csv", mutate)
        with self.assertRaisesRegex(verify.VerificationError, "shared real-LoRA initial loss"):
            verify.verify_real_lora(self.root)

    def test_changed_real_pair_fails_without_hash_check(self):
        folder = self.copy_family("real_lora")
        self.change_csv(folder / "paired_deltas.csv", lambda rows: rows[0].update(
            candidate_minus_reference_final_val_loss="0.5"))
        with self.assertRaisesRegex(verify.VerificationError, "paired final_val_loss"):
            verify.verify_real_lora(self.root)

    def test_missing_real_run_fails_without_hash_check(self):
        folder = self.copy_family("real_lora")
        self.change_csv(folder / "all_results.csv", lambda rows: rows.pop())
        with self.assertRaisesRegex(verify.VerificationError, "11 runs"):
            verify.verify_real_lora(self.root)

    def test_wrong_transformer_cost_fails_without_hash_check(self):
        folder = self.copy_family("transformer")
        self.change_csv(folder / "run_level_deltas_exact_cost.csv", lambda rows: rows[0].update(reference_cost="1"))
        with self.assertRaisesRegex(verify.VerificationError, "reference_cost"):
            verify.verify_transformer(self.root)

    def test_nonfinite_transformer_loss_fails_without_hash_check(self):
        folder = self.copy_family("transformer")
        self.change_csv(folder / "run_level_budget_results.csv", lambda rows: rows[0].update(final_val_loss="nan"))
        with self.assertRaisesRegex(verify.VerificationError, "Non-finite"):
            verify.verify_transformer(self.root)

    def test_negative_transformer_budget_fails_without_hash_check(self):
        folder = self.copy_family("transformer")
        self.change_csv(folder / "run_level_budget_results.csv", lambda rows: rows[0].update(requested_budget="-1"))
        with self.assertRaisesRegex(verify.VerificationError, "requested budget"):
            verify.verify_transformer(self.root)


if __name__ == "__main__":
    unittest.main()
