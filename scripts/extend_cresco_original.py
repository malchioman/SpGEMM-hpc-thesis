#!/usr/bin/env python3
"""Extend tesi-003 rep-01, excluding only Reddit squared.

Uses frozen original executables and experiment scripts; never builds, downloads,
regenerates matrices, or changes existing jobs. Python standard library only.
Defaults to 36/64/121 nodes; --nodes selects one separate extension (also 100).
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

NODES = (36, 64, 121)
SUPPORTED_NODES = (1, 4, 9, 16, 36, 64, 100, 121)
MAIN = ("strong_scaling", "permutation", "rectangular")
RELATIVE_OUTPUT = "results/tesi-003/extension-036-064-121"
RUNTIME_FILES = (
    "scripts/lib/experiments.py", "scripts/matrices.py", "scripts/experiments.json",
    "scripts/matrices_catalog.json", "scripts/collect_machine.sh",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root):
    manifest = read(root / "manifest.json")
    for relative, expected in manifest["sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen file changed: " + relative)
    return manifest


def runner(root):
    sys.path.insert(0, str(root / "metadata/runtime/scripts/lib"))
    import experiments as e
    if Path(e.__file__).resolve() != (root / "metadata/runtime/scripts/lib/experiments.py").resolve():
        raise ValueError("An unrelated experiment runner was already imported")
    original_cases = e.cases

    def selected_cases(cfg, action):
        return [case for case in original_cases(cfg, action)
                if not (action == "strong_scaling" and case[2].name == "reddit.mtx"
                        and case[3].name == "reddit.mtx")]

    e.cases = selected_cases
    return e


def check_inputs(root, cfg):
    # Matrix data itself is hashed by the original runner on the compute nodes.
    for frozen in (root / "metadata/matrices").glob("*.json"):
        if read(frozen) != read(cfg.matrix_dir / "metadata" / frozen.name):
            raise ValueError("Matrix provenance differs from the original campaign: " + frozen.name)


def batch_script(root, job, profile):
    n = job["nodes"]
    target = root / job["target"]
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", "#SBATCH --job-name=" + job["name"],
             f"#SBATCH --nodes={n}", f"#SBATCH --ntasks={2*n}",
             "#SBATCH --ntasks-per-node=2", "#SBATCH --cpus-per-task=64",
             "#SBATCH --mem=450000M", "#SBATCH --time=24:00:00",
             "#SBATCH --exclusive", "#SBATCH --hint=nomultithread",
             "#SBATCH --output=" + q(str(target / "slurm-%j.out")),
             "#SBATCH --error=" + q(str(target / "slurm-%j.err"))]
    for key in ("partition", "account"):
        if profile.get(key):
            lines.append(f"#SBATCH --{key}={profile[key]}")
    lines += ["set -euo pipefail", "cd -- " + q(str(root / "metadata/runtime"))]
    if profile.get("module_init"):
        lines.append("source " + q(profile["module_init"]))
    if profile.get("modules"):
        lines.append("module load " + shlex.join(profile["modules"]))
    lines += ["job_dir=" + q(str(target / "nodes")) + '/"${SLURM_JOB_ID}"',
              'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1',
              "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > \"$job_dir/exit-code.txt\"' EXIT",
              'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
              '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" '
              + q(str(root / "metadata/legacy-bin")),
              shlex.join([profile["python"], str(root / "metadata/launcher.py"),
                          "run", "--root", str(root), "--nodes", str(n),
                          "--repetition", str(job.get("repetition", 1))])]
    return "\n".join(lines) + "\n"


def prepare(repo, root, nodes=NODES, repetitions=1, campaign="tesi003-ext", follow_extensions=False):
    if (not nodes or len(set(nodes)) != len(nodes) or set(nodes) - set(SUPPORTED_NODES)
            or not 1 <= repetitions <= 10 or not re.fullmatch(r"[a-zA-Z0-9_-]+", campaign)):
        raise ValueError("Invalid nodes, repetition count (1-10), or campaign name")
    if root.exists():
        manifest = verify(root)
        planned = [(r, n) for r in range(1, repetitions + 1) for n in nodes]
        actual = [(j.get("repetition", 1), j["nodes"]) for j in manifest["jobs"]]
        if (actual != planned or manifest.get("campaign", "tesi003-ext") != campaign
                or manifest.get("follow_extensions", False) != follow_extensions):
            raise ValueError("This output contains a different node plan; use a separate output directory")
        print("Already prepared: " + str(root))
        return
    original = repo / "results/tesi-003/definitiva-001/metadata"
    recovery = repo / "results/tesi-003/recovery-reddit-mem01/metadata"
    config = read(original / "plan/tesi003-r01-n16.json")
    profile = read(original / "profile.json")
    required = dict(ranks_per_node=2, threads=63, cpus_per_rank=64, runs=1,
                    campaign_repeats=1, warmup=0, repeats=1, trials=1,
                    schedule="guided", chunk=64, validate=False, seed=42, timeout=0)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Original measurement settings differ from the expected first repetition")
    if (profile["partition"] != "cresco8_cpu" or profile["memory_per_node_mb"] != 450000
            or profile["walltime"] != "24:00:00" or not profile["exclusive"]):
        raise ValueError("Original allocation profile differs from the expected configuration")
    hashes = read(recovery / "legacy-sha256.json")
    if set(hashes) != set(config["variants"]) or len(hashes) != 8:
        raise ValueError("Expected all eight original executables")
    for variant, expected in hashes.items():
        executable = recovery / "legacy-bin" / variant
        if digest(executable) != expected or not os.access(executable, os.X_OK):
            raise ValueError("Original executable missing, changed, or not executable: " + str(executable))
    if not Path(profile["python"]).is_file():
        raise ValueError("Original Python environment missing: " + profile["python"])

    # Extract only known regular files; do not unpack arbitrary archive paths.
    runtime = root / "metadata/runtime"
    with tarfile.open(original / "source.tar.gz", "r:gz") as archive:
        for name in RUNTIME_FILES:
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError("Expected a regular source file: " + name)
            destination = runtime / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
    for name in ("source.tar.gz", "profile.json", "CMakeCache.txt", "git-commit.txt", "git-diff.patch"):
        shutil.copy2(original / name, root / "metadata" / name)
    shutil.copytree(original / "matrices", root / "metadata/matrices")
    shutil.copytree(recovery / "legacy-bin", root / "metadata/legacy-bin")
    for variant, expected in hashes.items():
        if digest(root / "metadata/legacy-bin" / variant) != expected:
            raise ValueError("Executable changed while copying: " + variant)
    shutil.copy2(original / "CMakeCache.txt", root / "metadata/legacy-bin/CMakeCache.txt")
    shutil.copy2(original / "plan/tesi003-r01-n16.json", root / "metadata/original-config.json")
    shutil.copy2(Path(__file__), root / "metadata/launcher.py")
    e = runner(root)
    jobs = []
    for repetition in range(1, repetitions + 1):
        # Same rotation as the original ten-repetition campaign planner.
        offset = (repetition - 1) % len(config["variants"])
        order = config["variants"][offset:] + config["variants"][:offset]
        for n in nodes:
            stem = f"r{repetition:02d}-n{n}" if repetitions > 1 else f"n{n}"
            job = dict(nodes=n, repetition=repetition, name=f"{campaign}-r{repetition:02d}-n{n}",
                       target=f"rep-{repetition:02d}/n{n}", config=f"metadata/plan/{stem}.json",
                       script=f"metadata/plan/{stem}.sbatch",
                       submission_key=stem if repetitions > 1 else str(n))
            settings = dict(config, nodes=[n], variants=order, build_dir=str(root / "metadata/legacy-bin"),
                            results_dir=str(root / job["target"]))
            write(root / job["config"], settings)
            cfg = e.options(["all", "--config", str(root / job["config"])])
            check_inputs(root, cfg)
            counts = {action: len(e.cases(cfg, action)) * len(cfg.variants) for action in e.RUN_ACTIONS}
            if counts != dict(pilot=24, strong_scaling=72, permutation=32, rectangular=80):
                raise ValueError("Unexpected workload: " + str(counts))
            if not any(a.name == "reddit.mtx" and b.name == "reddit_restriction.mtx"
                       for _, _, a, b in e.cases(cfg, "rectangular")):
                raise ValueError("Reddit rectangular must be retained")
            for action in e.RUN_ACTIONS:
                for _, _, a, b in e.cases(cfg, action):
                    for path in (a, b):
                        if not path.is_file():
                            raise ValueError("Prepared input missing: " + str(path))
            (root / job["target"]).mkdir(parents=True)
            write(root / job["script"], batch_script(root, job, profile))
            job["expected"] = counts
            jobs.append(job)
    # Include the copied executables, runner, configurations and scheduler scripts.
    sealed = {p.relative_to(root).as_posix(): digest(p)
              for p in sorted((root / "metadata").rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts}
    write(root / "manifest.json", dict(source_repo=str(repo), profile=profile, jobs=jobs,
          excluded=dict(action="strong_scaling", matrix_a="reddit.mtx", matrix_b="reddit.mtx"),
          repetition=1, repetitions=repetitions, campaign=campaign, follow_extensions=follow_extensions,
          original_binary_sha256=hashes, sha256=sealed))
    print(f"Prepared: {len(jobs)} jobs, 184 main observations + 24 pilot checks per job.")
    print("Original executables; Reddit rectangular retained. Results: " + str(root))
    if repetitions > 1:
        print(f"{len(nodes)} sequential lanes; at most {sum(nodes)} nodes for this campaign.")


def receipt_path(root, job):
    return root / "submission" / (job.get("submission_key", str(job["nodes"])) + ".json")


def extension_predecessors():
    output = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"],
                                      "-o", "%i|%j|%T"], text=True)
    result = {}
    for line in output.splitlines():
        job_id, name, _ = [s.strip() for s in line.split("|")]
        match = re.fullmatch(r"tesi003-ext-r01-n(\d+)", name)
        if match and job_id.isdigit():
            result.setdefault(int(match[1]), []).append(job_id)
    return result


def submit(root):
    manifest = verify(root)
    state = root / "submission"
    state.mkdir(exist_ok=True)
    remaining = []
    for job in manifest["jobs"]:
        receipt = receipt_path(root, job)
        if receipt.exists():
            record = read(receipt)
            if not record.get("job_id"):
                raise ValueError("Uncertain earlier submission; inspect the queue before retrying: " + str(receipt))
            print(f"Already submitted: {job['name']}, job={record['job_id']}")
        else:
            remaining.append((job, receipt))
    # Repetitions share the same allocation: check each node count once before submission.
    checked_nodes = set()
    for job, _ in remaining:
        if job["nodes"] not in checked_nodes:
            subprocess.run(["sbatch", "--test-only", str(root / job["script"])], check=True)
            checked_nodes.add(job["nodes"])
    predecessors = (extension_predecessors() if remaining and manifest.get("follow_extensions") else {})
    jobs_by_key = {(j.get("repetition", 1), j["nodes"]): j for j in manifest["jobs"]}
    for job, receipt in remaining:
        repetition = job.get("repetition", 1)
        dependency = None
        if repetition > 1:
            previous = jobs_by_key[(repetition - 1, job["nodes"])]
            completion = root / previous["target"] / "completion.json"
            # Completed predecessors may have aged out of Slurm by a resumed submission.
            completed = read(completion) if completion.exists() else {}
            if not (completed.get("complete") and completed.get("counts") == previous["expected"]):
                dependency = "afterok:" + read(receipt_path(root, previous))["job_id"]
        elif predecessors.get(job["nodes"]):
            dependency = "afterany:" + ":".join(predecessors[job["nodes"]])
        command = ["sbatch", "--parsable"]
        if dependency:
            command.append("--dependency=" + dependency)
        command.append(str(root / job["script"]))
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(dict(name=job["name"], state="submitting", command=command), stream)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("Unexpected sbatch response; inspect the queue: " + result.stdout)
        write(receipt, dict(name=job["name"], job_id=job_id, response=result.stdout, dependency=dependency))
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        print(f"SUBMITTED {job['name']} ranks={2*job['nodes']} job={job_id} dependency={dependency or '-'}", flush=True)


def run(root, nodes, repetition=1):
    manifest = verify(root)
    selected = [j for j in manifest["jobs"] if j["nodes"] == nodes and j.get("repetition", 1) == repetition]
    if len(selected) != 1:
        raise ValueError("Requested node count/repetition is absent or duplicated in this plan")
    job = selected[0]
    e = runner(root)
    args = ["all", "--config", str(root / job["config"])]
    cfg = e.options(args)
    env = os.environ
    if (not env.get("SLURM_JOB_ID") or int(env.get("SLURM_JOB_NUM_NODES", 0)) != nodes
            or int(env.get("SLURM_NTASKS", 0)) != nodes * cfg.ranks_per_node
            or int(env.get("SLURM_CPUS_PER_TASK", 0)) != cfg.cpus_per_rank):
        raise ValueError("Slurm allocation does not match the frozen configuration")
    counts = []
    for group in env.get("SLURM_TASKS_PER_NODE", "").split(","):
        match = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", group)
        if not match:
            raise ValueError("Missing or invalid SLURM_TASKS_PER_NODE")
        counts.extend([int(match[1])] * int(match[2] or 1))
    if counts != [cfg.ranks_per_node] * nodes:
        raise ValueError("Ranks per physical node differ from the original configuration")
    check_inputs(root, cfg)
    for action in e.RUN_ACTIONS:
        if any((e.output_root(cfg, action) / action).glob("*.tsv")):
            raise ValueError("Existing measurements: refusing to append a duplicate repetition")
    with (cfg.results_dir / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(job_id=env["SLURM_JOB_ID"]), stream)
    rc = e.main(args)
    actual = {action: sum(len(e.read_table(e.output_root(cfg, action) / action / (v + ".tsv"), e.FIELDS))
                          for v in cfg.variants) for action in e.RUN_ACTIONS}
    complete = actual == job["expected"] and rc == 0
    write(cfg.results_dir / "completion.json", dict(complete=complete, counts=actual,
                                                    expected=job["expected"], job_id=env["SLURM_JOB_ID"]))
    if not complete:
        raise ValueError("Incomplete workflow: " + str(actual))
    print("COMPLETE: 184/184 main observations; 24/24 pilot checks.", flush=True)


def status(root):
    manifest = read(root / "manifest.json")
    ids = []
    for job in manifest["jobs"]:
        receipt = receipt_path(root, job)
        job_id = read(receipt).get("job_id", "?") if receipt.exists() else "not submitted"
        if job_id.isdigit():
            ids.append(job_id)
        target = root / job["target"]
        counts = {}
        for action in ("pilot",) + MAIN:
            total = 0
            for table in (target / action).glob("*.tsv"):
                # A row being appended is not counted until its final newline.
                lines = table.read_text(encoding="utf-8").splitlines(keepends=True)
                rows = csv.DictReader([line for line in lines if line.endswith("\n")], delimiter="\t")
                total += sum(None not in row and all(v is not None for v in row.values()) for row in rows)
            counts[action] = total
        print(f"NODI={job['nodes']} RIP={job.get('repetition', 1)}/{manifest.get('repetitions', 1)} JOB={job_id}"
              f" | principali {sum(counts[a] for a in MAIN)}/184"
              f" | pilot {counts['pilot']}/24", flush=True)
        logs = list((target / "nodes").glob("*/launch.log"))
        if logs:
            lines = max(logs, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8", errors="replace").splitlines()
            last = next((line for line in reversed(lines) if any(line.startswith(a + ":") for a in MAIN)), None)
            print("  " + (last or (lines[-1] if lines else "Log still empty")), flush=True)
    if ids:
        subprocess.run(["sacct", "-X", "-j", ",".join(ids),
                        "--format=JobID,JobName%25,State%20,Elapsed,ExitCode"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "run", "status"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--nodes", type=int, choices=SUPPORTED_NODES,
                        help="One node count; uses its own extension-NNN directory unless --root is supplied")
    parser.add_argument("--node-counts", type=int, nargs="+", choices=SUPPORTED_NODES,
                        help="prepare only: explicit multi-node-count plan; requires --root")
    parser.add_argument("--repetitions", type=int, choices=range(1, 11), default=1,
                        help="prepare only: separate jobs per node count, chained with afterok")
    parser.add_argument("--campaign", default="tesi003-ext", help="prepare only: scheduler name prefix")
    parser.add_argument("--after-current-extensions", action="store_true",
                        help="prepare only: first job per lane follows active tesi003 extension jobs of that size")
    parser.add_argument("--repetition", type=int, default=1, help="run only: repetition to execute")
    args = parser.parse_args()
    if args.node_counts and args.nodes is not None:
        parser.error("Choose --node-counts or --nodes")
    custom_prepare = (args.node_counts or args.repetitions != 1 or args.campaign != "tesi003-ext"
                      or args.after_current_extensions)
    if custom_prepare and (args.action != "prepare" or args.root is None):
        parser.error("Multi-job planning options require prepare and an explicit --root")
    repo = args.repo.resolve()
    default_output = (f"results/tesi-003/extension-{args.nodes:03d}"
                      if args.nodes is not None else RELATIVE_OUTPUT)
    root = args.root.resolve() if args.root else repo / default_output
    if args.action == "prepare":
        prepare(repo, root, nodes=args.node_counts or ((args.nodes,) if args.nodes is not None else NODES),
                repetitions=args.repetitions, campaign=args.campaign,
                follow_extensions=args.after_current_extensions)
    elif args.action == "submit":
        submit(root)
    elif args.action == "status":
        status(root)
    else:
        if args.nodes is None:
            parser.error("run requires --nodes")
        run(root, args.nodes, args.repetition)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr, flush=True)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
