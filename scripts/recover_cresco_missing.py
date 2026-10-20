#!/usr/bin/env python3
"""Recover one frozen run_all repetition without modifying its results or binaries.

Only square reddit/Trident uses the separately labelled compact build. All other
missing observations use copies of the original binaries. Run through the bundled
prepare/submit script on CRESCO; --run executes inside an allocated compute job.
"""
import argparse
import contextlib
import copy
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import traceback

from lib import experiments as e
from lib import allocated_experiment
import prepare_sbatchman as planner

SUFFIX = "_compact_v1"
MAIN = ("strong_scaling", "permutation", "rectangular")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_case(action, case, variant):
    return action == "strong_scaling" and case[0] == "reddit" and variant.startswith("trident_")


def key(cfg, case, variant):
    return (variant, case[2].relative_to(cfg.matrix_dir).as_posix(),
            case[3].relative_to(cfg.matrix_dir).as_posix(), str(cfg.nodes[0]))


def row_key(variant, row):
    return variant, row["matrix_a"], row["matrix_b"], row["nodes"]


def expected(cfg, action):
    return {key(cfg, case, variant): (case, variant)
            for case in e.cases(cfg, action) for variant in cfg.variants}


def inventory(cfg, action, directory, original=False):
    """Reject duplicates, partial TSVs and resource mismatches before any launch."""
    planned, rows = expected(cfg, action), {}
    for path in sorted((directory / action).glob("*.tsv")):
        variant = path.stem
        for row in e.read_table(path, e.FIELDS):
            identity = row_key(variant, row)
            if identity not in planned or identity in rows:
                raise ValueError("Unexpected or duplicate observation: " + repr(identity))
            case, _ = planned[identity]
            required = dict(nodes=cfg.nodes[0], ranks=cfg.nodes[0] * cfg.ranks_per_node,
                            ranks_per_node=cfg.ranks_per_node, threads_per_rank=cfg.threads,
                            cpus_per_rank=cfg.cpus_per_rank, topology="physical",
                            omp_schedule=cfg.schedule, omp_chunk=cfg.chunk,
                            warmup=cfg.warmup, repeats=cfg.repeats, trials=cfg.trials,
                            validation="PASS" if cfg.validate else "SKIPPED")
            for field, value in required.items():
                if row[field] != str(value):
                    raise ValueError("Mismatched " + field + " in " + str(path))
            protocol = e.PROTOCOLS[variant]
            if not original and compact_case(action, case, variant):
                protocol += SUFFIX
            if row["benchmark_protocol"] != protocol:
                raise ValueError("Unexpected benchmark protocol in " + str(path))
            e.validate_measurements(row, variant.startswith("trident_"))
            rows[identity] = row
    return rows


def queue():
    output = subprocess.check_output(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T"], text=True)
    return [tuple(part.strip() for part in line.split("|")) for line in output.splitlines()]


def prepare(source, campaign, output, repetitions, nodes, compact_build):
    if output.exists():
        raise ValueError("Recovery directory already exists; use its manifest to resume: " + str(output))
    active = queue()
    profile = planner.load_profile("slurm", campaign / "metadata/profile.json")
    plans = []
    for rep in repetitions:
        for n in nodes:
            tag = f"tesi003-r{rep:02d}-n{n}"
            if any(name.startswith(tag + "-") for _, name, _ in active):
                raise ValueError("Original repetition still queued/running: " + tag)
            source_config = campaign / "metadata/plan" / (tag + ".json")
            cfg = e.options(["all", "--config", str(source_config)])
            if (cfg.nodes != [n] or cfg.runs != 1 or cfg.campaign_repeats != 1 or cfg.local_check
                    or cfg.warmup != 0 or cfg.repeats != 1 or cfg.trials != 1):
                raise ValueError("Recovery expects one observation per case in a frozen physical job")
            if (cfg.ranks_per_node, cfg.threads, cfg.cpus_per_rank) != (2, 63, 64):
                raise ValueError("Unexpected original resources; refusing to change the experiment")
            old = cfg.results_dir
            if not old.is_dir() or (old / ".experiments.lock").exists():
                raise ValueError("Original result directory missing or locked: " + str(old))
            data = {action: inventory(cfg, action, old, original=True) for action in MAIN}
            plans.append((rep, n, cfg, source_config, old, data))
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    binary_manifest = {}
    legacy = output / "metadata/legacy-bin"
    legacy.mkdir(parents=True)
    for rep, n, cfg, source_config, old, data in plans:
        target = output / f"rep-{rep:02d}" / f"n{n}"
        target.mkdir(parents=True)
        for variant in cfg.variants:
            src = cfg.build_dir / variant
            checksum = digest(src)
            if variant in binary_manifest and binary_manifest[variant] != checksum:
                raise ValueError("Original jobs use different binaries: " + variant)
            if variant not in binary_manifest:
                shutil.copy2(src, legacy / variant)
                binary_manifest[variant] = checksum
        settings = e.read_json(source_config)
        settings.update(results_dir=str(target), build_dir=str(legacy))
        config_path = target / "metadata/config.json"
        save(config_path, settings)
        copied, refreshed, origins = 0, 0, []
        for action, rows in data.items():
            cases = expected(cfg, action)
            for variant in cfg.variants:
                keep = []
                for identity, row in rows.items():
                    if identity[0] != variant:
                        continue
                    case, _ = cases[identity]
                    if compact_case(action, case, variant):
                        refreshed += 1  # needed when extending the same protocol to 9/16 nodes
                    else:
                        keep.append(row)
                        copied += 1
                        origins.append(dict(action=action, key=list(identity), origin="original"))
                if keep:
                    e.write_table(target / action / (variant + ".tsv"), e.FIELDS, keep)
        hashes = {str(p): digest(p) for action in MAIN for p in (old / action).glob("*.tsv")}
        save(target / "metadata/original-files.json", hashes)
        save(target / "metadata/origins.json", origins)
        save(target / "metadata/original-config.json", e.read_json(source_config))
        total = sum(len(expected(cfg, action)) for action in MAIN)
        job = dict(repetition=rep, nodes=n, config=str(config_path), original=str(old),
                   target=str(target), copied=copied, missing=total-copied,
                   refreshed_compact_cases=refreshed, expected=total,
                   name=f"tesi003-fix-r{rep:02d}-n{n}")
        jobs.append(job)
        print(f"n{n} rep-{rep:02d}: {copied}/{total} retained; {total-copied} to run; "
              f"{refreshed} replacements explicitly required by the compact protocol")
    save(output / "metadata/legacy-sha256.json", binary_manifest)
    manifest = dict(source_repo=str(source), campaign=str(campaign), output=str(output),
                    compact_build=str(compact_build), legacy_build=str(legacy),
                    profile=profile, jobs=jobs, protocol_suffix=SUFFIX)
    save(output / "manifest.json", manifest)
    return manifest


@contextlib.contextmanager
def protocol(variant, compact):
    before = e.PROTOCOLS[variant]
    if compact:
        e.PROTOCOLS[variant] = before + SUFFIX
    try:
        yield
    finally:
        e.PROTOCOLS[variant] = before


def run_one(cfg, action, case, variant, headers, compact_build, compact=False):
    single = copy.copy(cfg)
    single.variants = [variant]
    if compact:
        single.build_dir = compact_build
    with protocol(variant, compact):
        e.run_experiment(single, action, [case], headers)


def run(manifest, index):
    job = manifest["jobs"][index]
    cfg = e.options(["all", "--config", job["config"]])
    allocated_experiment.check_allocation(cfg, "slurm", os.environ)
    target = cfg.results_dir
    for path, checksum in e.read_json(target / "metadata/original-files.json").items():
        if digest(path) != checksum:
            raise ValueError("Original results changed after recovery planning: " + path)
    for path, checksum in e.read_json(Path(manifest["output"]) / "metadata/binaries-sha256.json").items():
        if digest(path) != checksum:
            raise ValueError("Recovery executable changed: " + path)
    with e.result_lock(target):
        present = {action: inventory(cfg, action, target) for action in MAIN}
        missing = [(action, case, variant) for action in MAIN
                   for identity, (case, variant) in expected(cfg, action).items()
                   if identity not in present[action]]
        print(f"Recovery: {len(missing)} remaining observations", flush=True)
        pilot = copy.copy(cfg)
        pilot.results_dir = target / "checks" / (os.environ["SLURM_JOB_ID"] + "-compact")
        pilot.variants = [v for v in cfg.variants if v.startswith("trident_")]
        work = [(action, [case]) for action, case, _ in missing]
        need_compact = any(compact_case(*item) for item in missing)
        if need_compact:
            work.append(("pilot", e.cases(pilot, "pilot")))
        headers = e.input_preflight(cfg, work) if work else {}
        compact_build = Path(manifest["compact_build"])
        if need_compact:
            # Correctness checks run on compute nodes, never on the login node.
            for case in e.cases(pilot, "pilot"):
                for variant in pilot.variants:
                    run_one(pilot, "pilot", case, variant, headers, compact_build, compact=True)
        failures, compact_failed = [], False
        for action, case, variant in missing:
            compact = compact_case(action, case, variant)
            if compact and compact_failed:
                failures.append(dict(action=action, variant=variant, matrix=case[0],
                                     error="Skipped after a compact reddit failure in this allocation"))
                continue
            try:
                run_one(cfg, action, case, variant, headers, compact_build, compact)
            except RuntimeError as error:
                # A failed matrix must not discard/skip all the independent later experiments.
                failures.append(dict(action=action, variant=variant, matrix=case[0], error=str(error)))
                compact_failed |= compact
                print("RECOVERY FAILURE: " + str(error), flush=True)
                save(target / "metadata/failures.json", failures)
        final = {action: inventory(cfg, action, target) for action in MAIN}
        count = sum(len(rows) for rows in final.values())
        save(target / "metadata/completion.json", dict(completed=count, expected=job["expected"],
             complete=count == job["expected"], failures=failures, job_id=os.environ["SLURM_JOB_ID"]))
        # Export derived views from the validated union, preserving each row's protocol.
        for variant in cfg.variants:
            rows = [row for action in MAIN for identity, row in final[action].items() if identity[0] == variant]
            if rows:
                e.write_table(target / "phase_analysis" / (variant + ".tsv"), e.PHASE_FIELDS,
                              [{k: row[k] for k in e.PHASE_FIELDS} for row in rows])
            square = [row for identity, row in final["strong_scaling"].items() if identity[0] == variant]
            for row in square:
                row.update({f"{prefix}_density": str(int(row[f"{prefix}_nnz"]) /
                           (int(row["a_rows" if prefix == "c" else f"{prefix}_rows"]) *
                            int(row["b_cols" if prefix == "c" else f"{prefix}_cols"])))
                            for prefix in ("a", "b", "c")})
            if square:
                e.write_table(target / "matrix_structure" / (variant + ".tsv"), e.STRUCTURE_FIELDS, square)
        print(f"Recovery finished: {count}/{job['expected']}", flush=True)
        return 0 if count == job["expected"] else 1


def script_for(manifest, index):
    job, profile = manifest["jobs"][index], manifest["profile"]
    n = job["nodes"]
    target = Path(job["target"])
    lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name={job['name']}", f"#SBATCH --nodes={n}",
             f"#SBATCH --ntasks={2*n}", "#SBATCH --ntasks-per-node=2", "#SBATCH --cpus-per-task=64",
             "#SBATCH --mem=450000M", "#SBATCH --time=24:00:00", "#SBATCH --exclusive",
             "#SBATCH --hint=nomultithread", "#SBATCH --output=" + str(target / "slurm-%j.out"),
             "#SBATCH --error=" + str(target / "slurm-%j.err")]
    for option in ("partition", "account"):
        if profile.get(option):
            lines.append(f"#SBATCH --{option}={profile[option]}")
    lines += ["set -euo pipefail", "cd -- " + shlex.quote(str(e.REPO))]
    if profile.get("module_init"):
        lines.append("source " + shlex.quote(profile["module_init"]))
    if profile["modules"]:
        lines.append("module load " + shlex.join(profile["modules"]))
    lines += ["job_dir=" + shlex.quote(str(target / "nodes")) + '/"${SLURM_JOB_ID}"',
              'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1',
              "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > \"$job_dir/exit-code.txt\"' EXIT",
              'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
              '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" '
              + shlex.quote(manifest["compact_build"]),
              shlex.join([profile["python"], str(Path(__file__).resolve()), "--manifest",
                          str(Path(manifest["output"]) / "manifest.json"), "--run", str(index)])]
    return "\n".join(lines) + "\n"


def submit(manifest):
    output = Path(manifest["output"])
    with e.result_lock(output / "metadata/submission"):
        submitted_path = output / "metadata/submitted.json"
        submitted = e.read_json(submitted_path) if submitted_path.exists() else {}
        active = queue()
        node_counts = {j["nodes"] for j in manifest["jobs"]}
        original = re.compile(r"^tesi003-r\d+-n(\d+)-tesi003-")
        held = [job for job, name, state in active if state == "PENDING" and
                (m := original.match(name)) and int(m[1]) in node_counts]
        if held:
            subprocess.run(["scontrol", "hold"] + held, check=True)
            save(output / "metadata/held-original-jobs.json", held)
        for i, job in enumerate(manifest["jobs"]):
            if str(i) in submitted:
                print("Already submitted: " + submitted[str(i)])
                continue
            if any(name == job["name"] for _, name, _ in active):
                raise ValueError("A matching recovery job already exists; inspect before resubmitting")
            script = Path(job["target"]) / "metadata/job.sh"
            with script.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(script_for(manifest, i))
            # Recovery can start on separate exclusive nodes while original jobs finish.
            # Only serialize multiple recovery repetitions at the same node count.
            predecessors = [submitted[str(k)] for k in range(i) if str(k) in submitted
                            and manifest["jobs"][k]["nodes"] == job["nodes"]]
            command = ["sbatch", "--parsable"]
            if predecessors:
                command.append("--dependency=afterany:" + ":".join(predecessors))
            ident = subprocess.check_output(command + [str(script)], text=True).strip().split(";")[0]
            if not ident.isdigit():
                raise ValueError("Unrecognized sbatch response: " + ident)
            submitted[str(i)] = ident
            save(submitted_path, submitted)
            print(f"Submitted {ident}: {job['name']}, dependencies={predecessors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nodes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repetitions", type=int, nargs="+", default=[1])
    parser.add_argument("--compact-build", type=Path)
    parser.add_argument("--manifest", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=int)
    group.add_argument("--submit", action="store_true")
    group.add_argument("--seal", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        manifest = e.read_json(args.manifest)
        if args.run is not None:
            return run(manifest, args.run)
        if args.submit:
            submit(manifest)
        elif args.seal:
            sealed = Path(manifest["output"]) / "metadata/binaries-sha256.json"
            if (sealed.parent / "submitted.json").exists():
                raise ValueError("Already submitted: binaries must remain immutable")
            files = [Path(manifest["legacy_build"]) / v for v in e.PROTOCOLS]
            files += [Path(manifest["compact_build"]) / v for v in e.PROTOCOLS if v.startswith("trident_")]
            save(sealed, {str(p): digest(p) for p in files})
        elif args.status:
            for job in manifest["jobs"]:
                cfg = e.options(["all", "--config", job["config"]])
                counts = {a: len(inventory(cfg, a, cfg.results_dir)) for a in MAIN}
                print(f"n{job['nodes']} rep-{job['repetition']:02d}: "
                      f"{sum(counts.values())}/{job['expected']} {counts}")
            path = Path(manifest["output"]) / "metadata/submitted.json"
            if path.exists():
                subprocess.run(["sacct", "-X", "-j", ",".join(e.read_json(path).values()),
                                "--format=JobID,JobName%32,State%22,Elapsed,ExitCode"], check=True)
        else:
            parser.error("Choose --run, --submit, --seal or --status")
    else:
        if not args.source_repo or not args.output or not args.compact_build:
            parser.error("Preparation needs --source-repo, --output and --compact-build")
        if (len(set(args.nodes)) != len(args.nodes) or set(args.nodes) - {1, 4, 9, 16}
                or len(set(args.repetitions)) != len(args.repetitions)
                or any(r < 1 or r > 10 for r in args.repetitions)):
            parser.error("Use unique nodes in 1,4,9,16 and repetitions in 1..10")
        source = args.source_repo.resolve()
        prepare(source, source / "results/tesi-003/definitiva-001", args.output.resolve(),
                args.repetitions, args.nodes, args.compact_build.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        traceback.print_exc()
        sys.exit(1)
