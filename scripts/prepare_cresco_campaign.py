#!/usr/bin/env python3
"""Prepare separate CRESCO jobs using the existing SbatchMan experiment runner.

Defaults are a feasibility run, not a statistically powered measurement campaign.
This tool writes plans only; submission remains an explicit SbatchMan command.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import prepare_sbatchman as planner


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--walltime", default="02:00:00")
    parser.add_argument("--memory-mb", type=int, default=450000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    planner.identifier(args.campaign)
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Choose a new output directory; existing results/plans are never overwritten")
    profile = planner.load_profile("slurm", args.profile)
    profile.update(walltime=args.walltime, memory_per_node_mb=args.memory_mb)
    # Validate modified values through the existing scheduler/profile checks below.
    if args.memory_mb < 1:
        raise ValueError("memory-mb must be positive")
    import re
    if not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", args.walltime) or not any(
            int(x) for x in args.walltime.split(":")):
        raise ValueError("walltime must be positive HH:MM:SS")
    defaults = planner.experiments.read_json(planner.REPO / "scripts/experiments.json")
    catalog = planner.experiments.read_json(planner.REPO / "scripts/matrices_catalog.json")
    names = [entry["id"] for entry in catalog["matrices"] if entry["tier"] == "all"]
    permutations = set(defaults["permutation_matrices"])
    files, configurations, jobs, expected = {}, {}, [], []
    for name in names:
        actions = ["strong_scaling", "rectangular"]
        if name in permutations:
            actions.insert(1, "permutation")
        for action in actions:
            cfg = planner.experiments.options([
                action, "--nodes", "1", "4", "--matrices", name,
                "--ranks-per-node", "2", "--threads", "4", "--cpus-per-rank", "5",
                "--runs", str(args.runs), "--warmup", str(args.warmup),
                "--repeats", str(args.repeats), "--trials", str(args.trials),
                "--timeout", str(args.timeout), "--no-validate",
                "--mpi-arg=--mca", "--mpi-arg=btl_vader_single_copy_mechanism",
                "--mpi-arg=none", "--results-dir", str(root)])
            if not args.dry_run:
                for _, _, a, b in planner.experiments.cases(cfg, action):
                    for path in (a, b, cfg.matrix_dir / "metadata" / (name + "_s42.cases.json")):
                        if not path.is_file():
                            raise ValueError("Missing prepared input: " + str(path))
                for variant in cfg.variants:
                    if not (cfg.build_dir / variant).is_file():
                        raise ValueError("Missing executable: " + variant)
            directory = root / "metadata" / "plan" / name / action
            plan = planner.make_plan("slurm", profile, cfg, args.campaign, directory)
            for config in plan.pop("configs.yaml")[profile["cluster_name"]]["configs"]:
                configurations[config["name"]] = config
            for job in plan.pop("jobs.yaml")["jobs"]:
                tag = "-".join((args.campaign, name, action, job["config"]))
                job["config_jobs"][0]["tag"] = tag
                jobs.append(job)
                expected.append(dict(tag=tag, matrix=name, action=action,
                                     config=job["config"], rows=args.runs * len(cfg.variants)))
            for filename, content in plan.items():
                if filename.endswith(".sh"):
                    block = (
                        "job_dir=" + shlex.quote(str(root / "nodes")) + '/"${SLURM_JOB_ID}"\n'
                        'mkdir -p "$job_dir"\n'
                        'printf "%s\\n" "$SLURM_JOB_ID" > "$job_dir/job-id.txt"\n'
                        'exec > >(tee "$job_dir/launch.log") 2>&1\n'
                        "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > \"$job_dir/exit-code.txt\"' EXIT\n"
                        'mpirun --mca btl_vader_single_copy_mechanism none '
                        '-np "$SLURM_JOB_NUM_NODES" --map-by ppr:1:node --bind-to none '
                        'bash scripts/collect_machine.sh "$job_dir" build\n')
                    if content.count("\nexec ") != 1:
                        raise ValueError("Unexpected wrapper format")
                    content = content.replace("\nexec ", "\n" + block, 1)
                files[directory / filename] = content
    plan_root = root / "metadata" / "plan"
    files[plan_root / "configs.yaml"] = {profile["cluster_name"]: dict(
        scheduler="slurm", configs=list(configurations.values()))}
    files[plan_root / "jobs.yaml"] = dict(cluster_name=profile["cluster_name"], sequential=True, jobs=jobs)
    files[root / "metadata" / "expected.json"] = expected
    files[root / "metadata" / "profile.json"] = profile
    print(f"{len(jobs)} sequential jobs; {sum(e['rows'] for e in expected)} expected observations")
    print("Results: " + str(root))
    if args.dry_run:
        print(json.dumps(expected, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    for path, content in files.items():
        save(path, content)
    for filename, command in (("git-commit.txt", ["git", "rev-parse", "HEAD"]),
                              ("git-status.txt", ["git", "status", "--short"]),
                              ("git-diff.patch", ["git", "diff", "--no-ext-diff", "HEAD"])):
        result = subprocess.run(command, cwd=planner.REPO, capture_output=True, check=True)
        (root / "metadata" / filename).write_bytes(result.stdout)
    save(root / "metadata" / "prepare_cresco_campaign.py", Path(__file__).read_text(encoding="utf-8"))
    for command in (["sbatchman", "set-cluster-name", profile["cluster_name"]],
                    ["sbatchman", "configure", "--file", str(plan_root / "configs.yaml"), "--overwrite"],
                    ["sbatchman", "launch", "--file", str(plan_root / "jobs.yaml")]):
        print(shlex.join(command))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
