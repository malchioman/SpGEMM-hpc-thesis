#!/usr/bin/env python3
"""Prepare ten run_all repetitions per topology in four sequential SbatchMan lanes.

No submission occurs here. Each allocation literally invokes scripts/run_all.sh.
One measured sample per MPI launch; ten launches per main case across allocations.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import prepare_sbatchman as planner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=planner.REPO / "results/tesi-001/definitiva-001")
    parser.add_argument("--nodes", nargs="+", type=int,
                        help="Override node counts without editing existing campaign settings")
    parser.add_argument("--after-jobs", nargs=4, type=int, metavar="JOB_ID",
                        help="Existing tail job for each of the four lanes, in lane order")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if root.exists():
        raise ValueError("Output already exists; do not overwrite or resubmit an existing plan")
    runner_args = ["all", "--config", str(planner.REPO / "scripts/experiments.cresco-thesis.json")]
    if args.nodes:
        runner_args += ["--nodes"] + [str(n) for n in args.nodes]
    cfg = planner.experiments.options(runner_args)
    if args.after_jobs and (any(n <= 0 for n in args.after_jobs) or len(set(args.after_jobs)) != 4):
        raise ValueError("after-jobs must contain four distinct positive job IDs")
    profile = planner.load_profile("slurm", args.profile)
    profile.update(walltime="24:00:00", memory_per_node_mb=450000, exclusive=True)
    if cfg.campaign_repeats != 10 or cfg.runs != 1 or cfg.timeout != 0:
        raise ValueError("Expected ten workflow repetitions, runs=1 and timeout=0")
    if not args.dry_run:
        for action in planner.experiments.RUN_ACTIONS:
            for name, _, a, b in planner.experiments.cases(cfg, action):
                for path in (a, b, cfg.matrix_dir / "metadata" / (name + "_s42.cases.json")):
                    if not path.is_file():
                        raise ValueError("Missing prepared input: " + str(path))
        for name in cfg.variants:
            if not (cfg.build_dir / name).is_file():
                raise ValueError("Missing executable: " + name)
    keys = planner.experiments.read_json(planner.REPO / "scripts/experiments.json")
    snapshot = {k: str(getattr(cfg, k)) if isinstance(getattr(cfg, k), Path)
                else getattr(cfg, k) for k in keys}
    files, configs, lanes, expected = {}, {}, [[] for _ in range(4)], []
    for repetition in range(10):
        # Rotate variant order across repetitions; record the exact order in each snapshot.
        order = cfg.variants[repetition % len(cfg.variants):] + cfg.variants[:repetition % len(cfg.variants)]
        for nodes in cfg.nodes:
            tag = "tesi001-r%02d-n%d" % (repetition + 1, nodes)
            target = root / ("rep-%02d" % (repetition + 1)) / ("n%d" % nodes)
            config_path = root / "metadata/plan" / (tag + ".json")
            wrapper_path = config_path.with_suffix(".sh")
            settings = dict(snapshot, nodes=[nodes], campaign_repeats=1,
                            results_dir=str(target), variants=order)
            files[config_path] = settings
            script = ["#!/usr/bin/env bash", "set -euo pipefail", "cd -- " + shlex.quote(str(planner.REPO))]
            if profile["module_init"]:
                script.append("source " + shlex.quote(profile["module_init"]))
            if profile["modules"]:
                script.append("module load " + shlex.join(profile["modules"]))
            script += ["export PYTHON=" + shlex.quote(profile["python"]),
                "job_dir=" + shlex.quote(str(target / "nodes")) + '/"${SLURM_JOB_ID}"',
                'mkdir -p "$job_dir"',
                'printf "%s\\n" "$SLURM_JOB_ID" > "$job_dir/job-id.txt"',
                'exec > >(tee "$job_dir/launch.log") 2>&1',
                "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > \"$job_dir/exit-code.txt\"' EXIT",
                'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
                '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" build',
                '"$PYTHON" -c ' + shlex.quote(
                    "import sys,os; sys.path.insert(0,'scripts/lib'); import experiments,allocated_experiment; "
                    "c=experiments.options(['all','--config',sys.argv[1]]); "
                    "allocated_experiment.check_allocation(c,'slurm',os.environ)") + " " + shlex.quote(str(config_path)),
                "bash scripts/run_all.sh --config " + shlex.quote(str(config_path))]
            files[wrapper_path] = "\n".join(script) + "\n"
            lane = len(expected) % 4
            config_name = "tesi001-n%d-r2-t4-c5" % nodes
            if args.after_jobs:
                config_name += "-lane%d-after%d" % (lane + 1, args.after_jobs[lane])
            if config_name not in configs:
                resource = planner.scheduler_config("slurm", profile, cfg, nodes, config_name)
                if args.after_jobs:
                    # SbatchMan supplies the later links in each sequential lane.
                    # The first submission inherits this dependency from its batch header.
                    resource["custom_headers"].append(
                        "#SBATCH --dependency=afterany:%d" % args.after_jobs[lane])
                configs[config_name] = resource
            lanes[lane].append(dict(config=config_name, config_jobs=[dict(
                tag=tag, command=shlex.join(["bash", str(wrapper_path)]))]))
            expected.append(dict(tag=tag, repetition=repetition + 1, nodes=nodes, lane=lane + 1,
                                 results_dir=str(target), expected_main_rows=192, expected_pilot_rows=24))
    files[root / "metadata/plan/configs.yaml"] = {profile["cluster_name"]: dict(scheduler="slurm", configs=list(configs.values()))}
    for index, jobs in enumerate(lanes, 1):
        files[root / ("metadata/plan/lane-%d.yaml" % index)] = dict(
            cluster_name=profile["cluster_name"], sequential=True, jobs=jobs)
    files[root / "metadata/expected.json"] = expected
    files[root / "metadata/profile.json"] = profile
    files[root / "metadata/lane-predecessors.json"] = args.after_jobs or []
    print("%d jobs: ten complete run_all passes at each node count %s." % (len(expected), cfg.nodes))
    print("Four sequential lanes: at most four jobs running simultaneously.")
    print("%d main observations + %d pilot checks expected. Per-launch timeout disabled." %
          (192 * len(expected), 24 * len(expected)))
    if args.after_jobs:
        print("Lane predecessors: " + str(args.after_jobs))
    print("Results: " + str(root))
    if args.dry_run:
        return 0
    root.mkdir(parents=True, exist_ok=False)
    for path, value in files.items():
        write(path, value)
    for name, cmd in (("git-commit.txt", ["git", "rev-parse", "HEAD"]),
                      ("git-status.txt", ["git", "status", "--short"]),
                      ("git-diff.patch", ["git", "diff", "--no-ext-diff", "HEAD"])):
        result = subprocess.run(cmd, cwd=planner.REPO, capture_output=True, check=True)
        (root / "metadata" / name).write_bytes(result.stdout)
    write(root / "metadata/prepare_cresco_run_all.py", Path(__file__).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
