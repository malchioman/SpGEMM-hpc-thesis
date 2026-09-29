#!/usr/bin/env python3
"""Fetch selected SuiteSparse/NERSC inputs and prepare reproducible SpGEMM pairs.

Python 3.8+ standard library only. Entries are streamed; permutation memory is O(n).
No benchmark executable or timing code is changed by this tool.
"""
import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]
INT_MAX = 2**31 - 1
DEFAULT_DOWNLOAD_TIMEOUT = 300
GENERATOR = "spgemm_inputs_v1"
UPSTREAM = ("https://github.com/HicrestLaboratory/Trident/blob/"
            "c37debaccc58b72859f1837f260900a40094848c/"
            "amg_matrices/multigrid_solver_matrix_generator.cpp")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_lines(stream):
    for line in stream:
        line = line.strip()
        if line and not line.startswith("%"):
            yield line


def read_header(stream):
    header = stream.readline().lower().split()
    if len(header) != 5 or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
        raise ValueError("expected a Matrix Market coordinate matrix")
    field, symmetry = header[3:]
    if field not in ("real", "integer", "pattern"):
        raise ValueError("complex matrices are not supported by the benchmarks")
    if symmetry not in ("general", "symmetric", "skew-symmetric", "hermitian"):
        raise ValueError("unsupported Matrix Market symmetry")
    dimensions = next(data_lines(stream), "").split()
    if len(dimensions) != 3:
        raise ValueError("invalid Matrix Market dimensions")
    rows, cols, entries = map(int, dimensions)
    if not (0 < rows <= INT_MAX and 0 < cols <= INT_MAX and 0 <= entries <= INT_MAX):
        raise ValueError("matrix exceeds the benchmark's signed 32-bit index/count range")
    if symmetry != "general" and rows != cols:
        raise ValueError("symmetric storage requires a square matrix")
    return dict(rows=rows, cols=cols, entries=entries, field=field, symmetry=symmetry)


def header_of(path):
    with Path(path).open(encoding="ascii") as stream:
        return read_header(stream)


def entries_of(path):
    """Yield zero-based entries, explicitly expanding symmetry like the C++ reader."""
    with Path(path).open(encoding="ascii") as stream:
        header = read_header(stream)
        count = 0
        for line in data_lines(stream):
            fields = line.replace("D", "E").replace("d", "e").split()
            expected = 2 if header["field"] == "pattern" else 3
            if len(fields) != expected:
                raise ValueError("invalid Matrix Market entry")
            row, col = int(fields[0]) - 1, int(fields[1]) - 1
            value = 1.0 if expected == 2 else float(fields[2])
            if not (0 <= row < header["rows"] and 0 <= col < header["cols"]):
                raise ValueError("Matrix Market index outside dimensions")
            if not math.isfinite(value):
                raise ValueError("non-finite matrix value")
            count += 1
            if count > header["entries"]:
                raise ValueError("more entries than declared in Matrix Market header")
            yield row, col, value
            if header["symmetry"] != "general" and row != col:
                yield col, row, -value if header["symmetry"] == "skew-symmetric" else value
        if count != header["entries"]:
            raise ValueError("truncated Matrix Market input")


def inspect_matrix(path):
    metadata = header_of(path)
    metadata["expanded_entries"] = sum(1 for _ in entries_of(path))
    if metadata["expanded_entries"] > INT_MAX:
        raise ValueError("expanded matrix exceeds the benchmark's 32-bit count range")
    metadata["sha256"] = sha256(path)
    return metadata


def extract_matrix(archive, name, target):
    # Extract only the exact regular file: never follow archive links or paths.
    wanted = name + "/" + name + ".mtx"
    with tarfile.open(archive, "r:gz") as source:
        matches = [m for m in source.getmembers()
                   if (m.name[2:] if m.name.startswith("./") else m.name) == wanted]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("archive does not contain one regular " + wanted)
        with source.extractfile(matches[0]) as reader, Path(target).open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)


def download(url, target, timeout=DEFAULT_DOWNLOAD_TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": "SpGEMM-thesis-matrices/1"})
    # This limits blocking socket operations, not the total transfer duration.
    with urllib.request.urlopen(request, timeout=timeout or None) as response, Path(target).open("wb") as output:
        if not response.geturl().startswith("https://"):
            raise ValueError("download redirected away from HTTPS")
        shutil.copyfileobj(response, output, length=1024 * 1024)


def source_of(entry):
    if "source" in entry:
        source = entry["source"]
        if source["format"] not in ("mtx", "tar.gz"):
            raise ValueError("unsupported download format")
        if not source["urls"] or any(not url.startswith("https://") for url in source["urls"]):
            raise ValueError("matrix sources must use HTTPS")
        return source
    suffix = "{}/{}.tar.gz".format(entry["group"], entry["id"])
    return dict(format="tar.gz", urls=[
        "https://sparse-files.engr.tamu.edu/MM/" + suffix,
        "https://www.cise.ufl.edu/research/sparse/MM/" + suffix],
        page="https://sparse.tamu.edu/{}/{}".format(entry["group"], entry["id"]))


def check_catalog_metadata(entry, metadata):
    # Some paper counts refer to stored entries in symmetric Matrix Market files.
    kind = entry.get("nnz_kind", "expanded")
    if kind not in ("stored", "expanded"):
        raise ValueError("nnz_kind must be stored or expanded")
    count = metadata["entries" if kind == "stored" else "expanded_entries"]
    if (metadata["rows"], metadata["cols"], count) != (entry["rows"], entry["cols"], entry["nnz"]):
        raise ValueError("downloaded dimensions/{} entries differ from the catalogue".format(kind))


def fetch(entry, root, timeout=DEFAULT_DOWNLOAD_TIMEOUT):
    name = entry["id"]
    matrix = root / (name + ".mtx")
    record_path = root / "metadata" / (name + ".source.json")
    if matrix.exists() or record_path.exists():
        if not matrix.is_file() or not record_path.is_file():
            raise ValueError("incomplete input: expected both {} and {}; restore or move the incomplete files".format(
                matrix, record_path))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["sha256"] != sha256(matrix):
            raise ValueError("existing input changed: " + str(matrix))
        check_catalog_metadata(entry, record)
        print("Already verified:", matrix)
        return
    record_path.parent.mkdir(parents=True, exist_ok=True)
    source = source_of(entry)
    urls = source["urls"]
    with tempfile.TemporaryDirectory(prefix=".fetch-", dir=root) as work:
        work = Path(work)
        archive = work / ("download." + source["format"])
        for index, url in enumerate(urls):
            print("Downloading:", url, flush=True)
            try:
                download(url, archive, timeout=timeout)
                break
            except OSError as error:
                if index + 1 == len(urls):
                    raise
                print("Server unavailable; trying the next source:", error, flush=True)
        output = work / "ready"
        output.mkdir()
        extracted = output / matrix.name
        downloaded_hash = sha256(archive)
        if source["format"] == "tar.gz":
            extract_matrix(archive, name, extracted)
        else:
            archive.rename(extracted)
        metadata = inspect_matrix(extracted)
        check_catalog_metadata(entry, metadata)
        metadata.update(source_url=url, source_page=source["page"],
            download_format=source["format"], download_sha256=downloaded_hash,
            catalog_nnz=entry["nnz"], catalog_nnz_kind=entry.get("nnz_kind", "expanded"),
            fetched_utc=datetime.now(timezone.utc).isoformat())
        if source["format"] == "tar.gz":
            metadata["archive_sha256"] = downloaded_hash
        # Publish the completion record last; an interrupted pair is never reused.
        write_json(output / record_path.name, metadata)
        extracted.rename(matrix)
        (output / record_path.name).rename(record_path)
    print("Verified:", matrix)
    print("  Stored entries: {}; expanded entries: {}".format(
        metadata["entries"], metadata["expanded_entries"]))


def permutation(n, seed):
    """Old-index -> new-index, Fisher-Yates + specified SplitMix64/rejection sampling."""
    mask = 2**64 - 1
    state = seed
    values = array("I", range(n))
    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = 2**64 - 2**64 % bound
        while True:
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            if z < limit:
                break
        j = z % bound
        values[i], values[j] = values[j], values[i]
    return values


def permute_matrix(source, output, metadata, seed):
    n = metadata["rows"]
    if n != metadata["cols"]:
        raise ValueError("P A P^T requires a square matrix")
    mapping = permutation(n, seed)
    count = 0
    with Path(output).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%%MatrixMarket matrix coordinate real general\n")
        stream.write("% simultaneous row/column permutation; splitmix64_fisher_yates_v1 seed={}\n".format(seed))
        stream.write("{} {} {}\n".format(n, n, metadata["expanded_entries"]))
        for row, col, value in entries_of(source):
            stream.write("{} {} {:.17g}\n".format(mapping[row] + 1, mapping[col] + 1, value))
            count += 1
    if count != metadata["expanded_entries"]:
        raise ValueError("permutation entry count mismatch")


def restriction_matrix(n, output):
    """R: n x (2n-1), 3n-2 entries; same 1D weights as the upstream generator."""
    if not (0 < n and 2 * n - 1 <= INT_MAX and 3 * n - 2 <= INT_MAX):
        raise ValueError("restriction operator exceeds the benchmark index/count range")
    with Path(output).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("%%MatrixMarket matrix coordinate real general\n")
        stream.write("% synthetic 1D full-weighting restriction; endpoint weights are not renormalized\n")
        stream.write("{} {} {}\n".format(n, 2 * n - 1, 3 * n - 2))
        for row in range(n):
            if row > 0:
                stream.write("{} {} 0.25\n".format(row + 1, 2 * row))
            stream.write("{} {} 0.5\n".format(row + 1, 2 * row + 1))
            if row + 1 < n:
                stream.write("{} {} 0.25\n".format(row + 1, 2 * row + 2))


def prepare(entry, root, seed):
    name = entry["id"]
    source = root / (name + ".mtx")
    metadata_dir = root / "metadata"
    original = json.loads((metadata_dir / (name + ".source.json")).read_text(encoding="utf-8"))
    if sha256(source) != original["sha256"]:
        raise ValueError("source hash changed: " + str(source))
    manifest = metadata_dir / (name + "_s" + str(seed) + ".cases.json")
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record["generator"] != GENERATOR or record["seed"] != seed or record["matrix"] != name:
            raise ValueError("existing preparation uses a different generator")
        for data in record["files"].values():
            if sha256(metadata_dir / data["path"]) != data["sha256"]:
                raise ValueError("prepared input changed; use a new output directory")
        print(manifest)
        return
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as work:
        work = Path(work)
        permuted = work / (name + "_permuted_s" + str(seed) + ".mtx")
        restriction = work / (name + "_restriction.mtx")
        permute_matrix(source, permuted, original, seed)
        restriction_matrix(original["cols"], restriction)
        files = {"original": dict(original, path="../" + source.name)}
        outputs = [("permuted", permuted, Path("permuted") / permuted.name),
                   ("restriction", restriction, Path(restriction.name))]
        for key, path, relative in outputs:
            files[key] = dict(inspect_matrix(path), path="../" + relative.as_posix())
            target = root / relative
            if target.exists() and sha256(target) != files[key]["sha256"]:
                raise ValueError("prepared input changed; refusing to overwrite " + str(target))
        record = dict(schema_version=1, generator=GENERATOR, matrix=name, seed=seed,
                      permutation="splitmix64_fisher_yates_v1", restriction_source=UPSTREAM,
                      files=files, cases=[
                          dict(id=name + "_square", product="A*A", matrix_a="original", matrix_b="original"),
                          dict(id=name + "_permuted_s" + str(seed), product="(P*A*P^T)^2", matrix_a="permuted", matrix_b="permuted"),
                          dict(id=name + "_rectangular", product="A*R", matrix_a="original", matrix_b="restriction")])
        # R is independent of the seed and shared by all preparations of A.
        # Check every existing file before publishing any new file.
        for _, path, relative in outputs:
            target = root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
        write_json(manifest, record)
    print(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "fetch", "prepare"])
    parser.add_argument("--catalog", type=Path, default=REPO / "scripts/matrices_catalog.json")
    parser.add_argument("--root", type=Path, default=REPO / "bin/matrices",
                        help="matrix directory; permutations go in permuted/ (default: bin/matrices)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--tier", choices=["pilot", "core", "all"],
                           help="all selects the ten main matrices; pilot/core are auxiliary sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=DEFAULT_DOWNLOAD_TIMEOUT,
                        help="download socket timeout in seconds (default: 300); 0 disables it, not a total transfer limit")
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative; use 0 to disable it")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))["matrices"]
    for entry in catalog:
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", entry[k]) for k in ("id", "group")):
            parser.error("catalogue IDs and groups must be simple names")
    known = {entry["id"] for entry in catalog}
    if args.ids and (set(args.ids) - known):
        parser.error("unknown matrix IDs: " + ", ".join(sorted(set(args.ids) - known)))
    if args.action != "list" and not args.ids and not args.tier:
        parser.error("select --ids or --tier explicitly; large inputs are never downloaded implicitly")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    entries = [e for e in catalog if (not args.ids or e["id"] in args.ids)
               and (not args.tier or e["tier"] == args.tier)]
    for entry in entries:
        if args.action == "list":
            print(("{id:18} {tier:6} {rows:>9} x {cols:<9} nnz={nnz:<10} (" +
                   entry.get("nnz_kind", "expanded") + ") {role}").format(**entry))
        elif args.action == "fetch":
            fetch(entry, args.root.resolve(), timeout=args.timeout)
        else:
            prepare(entry, args.root.resolve(), args.seed)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print("Matrix preparation:", error, file=sys.stderr)
        sys.exit(1)
