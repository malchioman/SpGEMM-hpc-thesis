#!/usr/bin/env python3
"""Offline checks for matrix downloads and scientific input transformations."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matrices


def dense(path):
    shape = matrices.header_of(path)
    result = [[0.0] * shape["cols"] for _ in range(shape["rows"])]
    for i, j, v in matrices.entries_of(path):
        result[i][j] += v
    return result


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


class MatrixToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spgemm-experiment-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "fixture.mtx"
        (self.root / "metadata").mkdir()
        self.source.write_text("%%MatrixMarket matrix coordinate real symmetric\n"
                               "% a signed symmetric matrix, stored as one triangle\n"
                               "4 4 6\n1 1 2\n2 1 -3\n2 2 1\n3 2 4\n4 1 5\n4 4 -2\n",
                               encoding="ascii")
        matrices.write_json(self.root / "metadata/fixture.source.json", matrices.inspect_matrix(self.source))

    def prepare(self, seed=42):
        with contextlib.redirect_stdout(io.StringIO()):
            matrices.prepare({"id": "fixture"}, self.root, seed)
        return self.root / "metadata" / ("fixture_s" + str(seed) + ".cases.json")

    def test_permutation_preserves_matrix_and_squared_product(self):
        manifest = self.prepare()
        original = dense(self.source)
        permuted = dense(self.root / "permuted/fixture_permuted_s42.mtx")
        mapping = matrices.permutation(4, 42)
        self.assertEqual(sorted(mapping), list(range(4)))
        expected = [[0.0] * 4 for _ in range(4)]
        squared = [[0.0] * 4 for _ in range(4)]
        product = multiply(original, original)
        for i in range(4):
            for j in range(4):
                expected[mapping[i]][mapping[j]] = original[i][j]
                squared[mapping[i]][mapping[j]] = product[i][j]
        self.assertEqual(permuted, expected)
        self.assertEqual(multiply(permuted, permuted), squared)
        self.assertEqual(matrices.inspect_matrix(self.source)["expanded_entries"],
                         matrices.inspect_matrix(self.root / "permuted/fixture_permuted_s42.mtx")["expanded_entries"])

    def test_permutation_reproducible_and_corrupt_cache_rejected(self):
        manifest = self.prepare()
        before = manifest.read_bytes()
        self.prepare()
        self.assertEqual(before, manifest.read_bytes())
        first = self.root / "again.mtx"
        matrices.permute_matrix(self.source, first, matrices.inspect_matrix(self.source), 42)
        self.assertEqual(first.read_bytes(), (self.root / "permuted/fixture_permuted_s42.mtx").read_bytes())
        (self.root / "permuted/fixture_permuted_s42.mtx").write_text("broken", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "prepared input changed"):
            self.prepare()

    def test_inputs_share_restriction_across_seeds_and_survive_relocation(self):
        manifest = self.prepare()
        restriction = self.root / "fixture_restriction.mtx"
        before = restriction.stat().st_mtime_ns
        self.prepare(seed=7)
        self.assertEqual(before, restriction.stat().st_mtime_ns)
        self.assertEqual({p.relative_to(self.root).as_posix() for p in self.root.rglob("*.mtx")}, {
            "fixture.mtx", "permuted/fixture_permuted_s42.mtx", "permuted/fixture_permuted_s7.mtx", "fixture_restriction.mtx"})
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "matrices"
            shutil.copytree(self.root, moved)
            relocated = moved / "metadata" / manifest.name
            record = json.loads(relocated.read_text())
            self.assertEqual(len(record["cases"]), 3)
            for data in record["files"].values():
                path = (relocated.parent / data["path"]).resolve()
                self.assertEqual(matrices.sha256(path), data["sha256"])
                expected_parent = moved / "permuted" if "_permuted_" in path.name else moved
                self.assertEqual(path.parent, expected_parent.resolve())

    def test_prepare_does_not_overwrite_conflicting_files(self):
        restriction = self.root / "fixture_restriction.mtx"
        restriction.write_text("existing user file", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.prepare()
        self.assertEqual(restriction.read_text(), "existing user file")
        self.assertFalse((self.root / "permuted/fixture_permuted_s42.mtx").exists())
        self.assertFalse((self.root / "metadata/fixture_s42.cases.json").exists())

    def test_restriction_matches_full_weighting_stencil_and_endpoints(self):
        output = self.root / "restriction.mtx"
        matrices.restriction_matrix(3, output)
        self.assertEqual(dense(output), [[.5, .25, 0, 0, 0],
                                         [0, .25, .5, .25, 0],
                                         [0, 0, 0, .25, .5]])
        self.assertEqual(matrices.inspect_matrix(output)["expanded_entries"], 7)
        matrices.restriction_matrix(1, output)
        self.assertEqual(dense(output), [[.5]])
        with self.assertRaises(ValueError):
            matrices.restriction_matrix(matrices.INT_MAX, output)

    def test_skew_symmetry_and_pattern_inputs(self):
        for kind, body, expected in [
            ("real skew-symmetric", "2 1 3\n3 2 -4\n", [[0, -3, 0], [3, 0, 4], [0, -4, 0]]),
            ("pattern symmetric", "2 1\n3 2\n", [[0, 1, 0], [1, 0, 1], [0, 1, 0]])]:
            self.source.write_text("%%MatrixMarket matrix coordinate " + kind + "\n3 3 2\n" + body,
                                   encoding="ascii")
            self.assertEqual(dense(self.source), expected)
            output = self.root / "permuted.mtx"
            matrices.permute_matrix(self.source, output, matrices.inspect_matrix(self.source), 42)
            mapping = matrices.permutation(3, 42)
            observed = dense(output)
            for i in range(3):
                for j in range(3):
                    self.assertEqual(observed[mapping[i]][mapping[j]], expected[i][j])

    def test_malformed_entries_rejected(self):
        for body in ["1 1 nan\n", "3 1 2\n", "", "1 1 2\n1 2 3\n"]:
            self.source.write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n" + body,
                                   encoding="ascii")
            with self.assertRaises(ValueError):
                matrices.inspect_matrix(self.source)
        self.source.write_text("%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n1 1 3\n",
                               encoding="ascii")
        with self.assertRaisesRegex(ValueError, "diagonal must be zero"):
            matrices.inspect_matrix(self.source)

    def test_archive_only_extracts_named_regular_matrix(self):
        archive = self.root / "matrix.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(self.source, arcname="fixture/fixture.mtx")
            unsafe = tarfile.TarInfo("../escape.txt")
            unsafe.size = 3
            stream.addfile(unsafe, io.BytesIO(b"bad"))
        output = self.root / "extracted.mtx"
        matrices.extract_matrix(archive, "fixture", output)
        self.assertEqual(output.read_bytes(), self.source.read_bytes())
        self.assertFalse((self.root.parent / "escape.txt").exists())
        with tarfile.open(archive, "w:gz") as stream:
            link = tarfile.TarInfo("fixture/fixture.mtx")
            link.type = tarfile.SYMTYPE
            link.linkname = str(self.source)
            stream.addfile(link)
        with self.assertRaises(ValueError):
            matrices.extract_matrix(archive, "fixture", output)

    def test_all_catalog_contains_all_ten_download_sources(self):
        catalog = json.loads((matrices.REPO / "scripts/matrices_catalog.json").read_text())["matrices"]
        all_inputs = [entry for entry in catalog if entry["tier"] == "all"]
        expected = {"HV15R", "mouse_gene", "archaea", "eukarya", "isolates_subgraph4",
                    "isolates_subgraph5", "cage15", "uniparc", "reddit", "dielFilterV3real"}
        self.assertEqual(len(all_inputs), 10)
        self.assertEqual({entry["id"] for entry in all_inputs}, expected)
        for entry in all_inputs:
            self.assertTrue(entry["paper"])
            self.assertTrue(matrices.source_of(entry)["urls"])
        self.assertEqual({e["id"] for e in all_inputs if e.get("nnz_kind") == "stored"},
                         {"isolates_subgraph4", "isolates_subgraph5", "reddit"})

    def test_direct_fetch_preserves_symmetric_storage_and_reuses_verified_input(self):
        entry = dict(id="direct", rows=4, cols=4, nnz=6, nnz_kind="stored",
                     source=dict(format="mtx", urls=["https://example.org/input.mtx"],
                                 page="https://example.org/"))
        body = self.source.read_bytes()
        with mock.patch.object(matrices, "download", side_effect=lambda url, path, timeout: path.write_bytes(body)) as fetcher:
            with contextlib.redirect_stdout(io.StringIO()):
                matrices.fetch(entry, self.root, timeout=0)
                matrices.fetch(entry, self.root, timeout=0)
            self.assertEqual(fetcher.call_count, 1)
            self.assertEqual(fetcher.call_args.kwargs["timeout"], 0)
        output = self.root / "direct.mtx"
        self.assertEqual(output.read_bytes(), body)
        record = json.loads((self.root / "metadata/direct.source.json").read_text())
        self.assertEqual((record["entries"], record["expanded_entries"]), (6, 9))
        self.assertEqual(record["download_sha256"], record["sha256"])
        self.assertEqual(record["catalog_nnz_kind"], "stored")
        self.assertNotIn("archive_sha256", record)
        with self.assertRaisesRegex(ValueError, "differ from the catalogue"):
            matrices.fetch(dict(entry, nnz=5), self.root)
        output.write_bytes(body + b"% changed\n")
        with self.assertRaisesRegex(ValueError, "existing input changed"):
            matrices.fetch(entry, self.root)

    def test_fetch_rejects_wrong_count_without_publishing_partial_input(self):
        entry = dict(id="bad", rows=4, cols=4, nnz=6, nnz_kind="expanded",
                     source=dict(format="mtx", urls=["https://example.org/input.mtx"],
                                 page="https://example.org/"))
        with mock.patch.object(matrices, "download", side_effect=lambda url, path, timeout: path.write_bytes(self.source.read_bytes())):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "expanded entries differ"):
                matrices.fetch(entry, self.root)
        self.assertFalse((self.root / "bad.mtx").exists())
        self.assertEqual(list(self.root.glob(".fetch-*")), [])

    def test_fetch_does_not_overwrite_an_input_without_metadata(self):
        before = self.source.read_bytes()
        (self.root / "metadata/fixture.source.json").unlink()
        with mock.patch.object(matrices, "download") as fetcher:
            with self.assertRaisesRegex(ValueError, "incomplete input"):
                matrices.fetch(dict(id="fixture", rows=4, cols=4, nnz=9), self.root)
            fetcher.assert_not_called()
        self.assertEqual(self.source.read_bytes(), before)

    def test_suitesparse_fetch_falls_back_and_records_archive_hash(self):
        archive = self.root / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(self.source, arcname="downloaded/downloaded.mtx")
        calls = []
        def fake_download(url, path, timeout):
            self.assertEqual(timeout, 120)
            calls.append(url)
            if len(calls) == 1:
                raise OSError("primary unavailable")
            path.write_bytes(archive.read_bytes())
        entry = dict(id="downloaded", group="fixture", rows=4, cols=4, nnz=9)
        with mock.patch.object(matrices, "download", side_effect=fake_download):
            with contextlib.redirect_stdout(io.StringIO()):
                matrices.fetch(entry, self.root, timeout=120)
        self.assertEqual(len(calls), 2)
        record = json.loads((self.root / "metadata/downloaded.source.json").read_text())
        self.assertEqual(record["source_url"], calls[-1])
        self.assertEqual(record["archive_sha256"], matrices.sha256(archive))
        self.assertEqual(record["download_sha256"], record["archive_sha256"])

    def test_download_timeout_preserves_bytes_and_can_disable_socket_deadline(self):
        url = "https://example.org/input.mtx"
        body = self.source.read_bytes()
        for timeout, expected in ((0, None), (120, 120), (matrices.DEFAULT_DOWNLOAD_TIMEOUT, 300)):
            with self.subTest(timeout=timeout):
                response = io.BytesIO(body)
                response.geturl = lambda: url
                output = self.root / "download.mtx"
                with mock.patch.object(matrices.urllib.request, "urlopen", return_value=response) as opener:
                    matrices.download(url, output, timeout=timeout)
                self.assertEqual(opener.call_args.kwargs["timeout"], expected)
                self.assertEqual(output.read_bytes(), body)

    def test_changed_source_rejected_before_preparation(self):
        self.prepare()
        self.source.write_text(self.source.read_text().replace("1 1 2", "1 1 3"), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            matrices.prepare({"id": "fixture"}, self.root, 42)


if __name__ == "__main__":
    unittest.main()
