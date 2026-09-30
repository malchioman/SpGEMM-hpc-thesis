#!/usr/bin/env python3
"""Prepare SbatchMan jobs for Slurm or OpenPBS/PBS Pro without submitting them."""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

from lib import experiments

REPO = experiments.REPO
PROFILES = REPO / "scripts/sbatchman"


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Use only letters, digits, underscores and hyphens in names: " + value)
    return value


def load_profile(scheduler, path):
    profile = experiments.read_json(PROFILES / (scheduler + ".example.json"))
    if path:
        supplied = experiments.read_json(path)
        unknown = set(supplied) - set(profile)
        if unknown:
            raise ValueError("Unknown profile keys: " + ", ".join(sorted(unknown)))
        profile.update(supplied)
    identifier(profile["cluster_name"])
    if type(profile["memory_per_node_mb"]) is not int or profile["memory_per_node_mb"] < 1:
        raise ValueError("memory_per_node_mb must be a positive integer")
    if not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", profile["walltime"]) or not any(
            int(part) for part in profile["walltime"].split(":")):
        raise ValueError("walltime must be a positive HH:MM:SS duration")
    if type(profile["exclusive"]) is not bool:
        raise ValueError("exclusive must be true or false")
    if not isinstance(profile["modules"], list) or any(
            not isinstance(m, str) or not m or m.startswith("-") or "\n" in m or "\r" in m
            for m in profile["modules"]):
        raise ValueError("modules must be a list of module names")
    for key in ("partition", "queue", "account", "python", "module_init"):
        value = profile.get(key)
        if value is not None and (not isinstance(value, str) or not value or "\n" in value or "\r" in value):
            raise ValueError(key + " must be a nonempty single-line string or null")
    if not profile["python"]:
        raise ValueError("python must name the cluster Python executable")
    for key in ("partition", "queue", "account"):
        if profile.get(key) and not re.fullmatch(r"[A-Za-z0-9_.@,+-]+", profile[key]):
            raise ValueError("Invalid scheduler value for " + key)
    return profile


def scheduler_config(scheduler, profile, cfg, nodes, name):
    ranks, cpus = cfg.ranks_per_node, cfg.cpus_per_rank
    memory = profile["memory_per_node_mb"]
    if scheduler == "slurm":
        return dict(name=name, nodes=nodes, ntasks=nodes * ranks,
                    tasks_per_node=ranks, cpus_per_task=cpus, mem=str(memory) + "M",
                    time=profile["walltime"], partition=profile["partition"],
                    account=profile["account"], exclusive=profile["exclusive"],
                    custom_headers=["#SBATCH --hint=nomultithread"])
    # SbatchMan's PBS `cpus`/`mem` are job-wide. Use chunk resources instead,
    # and scatter chunks across physical hosts for Trident's node grid.
    headers = ["#PBS -l select=%d:ncpus=%d:mpiprocs=%d:ompthreads=%d:mem=%dmb" % (
        nodes, ranks * cpus, ranks, cfg.threads, memory),
        "#PBS -l place=scatter" + (":exclhost" if profile["exclusive"] else "")]
    if profile["account"]:
        headers.append("#PBS -A " + profile["account"])
    return dict(name=name, queue=profile["queue"], walltime=profile["walltime"],
                custom_headers=headers)


def make_plan(scheduler, profile, cfg, campaign, output, isolate_results=False):
    if cfg.local_check or cfg.hostfile or cfg.dry_run:
        raise ValueError("Scheduler jobs cannot use --local-check, --hostfile or runner --dry-run")
    keys = experiments.read_json(REPO / "scripts/experiments.json")
    snapshot = {key: str(getattr(cfg, key)) if isinstance(getattr(cfg, key), Path)
                else getattr(cfg, key) for key in keys}
    configs, jobs, files = [], [], {}
    for nodes in cfg.nodes:
        name = "n%d-r%d-t%d-c%d" % (nodes, cfg.ranks_per_node, cfg.threads, cfg.cpus_per_rank)
        configs.append(scheduler_config(scheduler, profile, cfg, nodes, name))
        # Keep the runner's output paths and append behavior unless isolation
        # was explicitly requested. The sequential jobs share its result lock.
        result_root = cfg.results_dir
        if isolate_results:
            result_root = result_root / "campaigns" / campaign / profile["cluster_name"] / name
        settings = dict(snapshot, nodes=[nodes], results_dir=str(result_root))
        files[name + ".json"] = settings
        command = [profile["python"], str(REPO / "scripts/lib/allocated_experiment.py"),
                   "--scheduler", scheduler, "--action", cfg.action,
                   "--config", str(output / (name + ".json"))]
        script = ["#!/usr/bin/env bash", "set -euo pipefail", "cd -- " + shlex.quote(str(REPO))]
        if profile["module_init"]:
            script.append("source " + shlex.quote(profile["module_init"]))
        if profile["modules"]:
            script.append("module load " + shlex.join(profile["modules"]))
        script.append("exec " + shlex.join(command))
        files[name + ".sh"] = "\n".join(script) + "\n"
        jobs.append(dict(config=name, config_jobs=[dict(
            tag=campaign + "-" + cfg.action, command=shlex.join(["bash", str(output / (name + ".sh"))]))]))
    files["configs.yaml"] = {profile["cluster_name"]: dict(scheduler=scheduler, configs=configs)}
    files["jobs.yaml"] = dict(cluster_name=profile["cluster_name"], sequential=True, jobs=jobs)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Pass experiment options after --, e.g. -- --nodes 1 4 --threads 8. "
        "Resource counts come from scripts/experiments.json unless overridden."))
    parser.add_argument("--scheduler", required=True, choices=("slurm", "pbs"))
    parser.add_argument("--profile", type=Path, help="cluster JSON profile; unspecified keys use the example")
    parser.add_argument("--campaign", required=True, help="new name for this set of jobs/results")
    parser.add_argument("--action", choices=experiments.RUN_ACTIONS + ("all",), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--isolate-results", action="store_true", help=(
        "separate results by campaign/cluster/resources; default: use the same output paths as run_all.sh"))
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing files")
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    identifier(args.campaign)
    profile = load_profile(args.scheduler, args.profile)
    forwarded = args.experiment_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    cfg = experiments.options([args.action] + forwarded)
    output = (args.output_dir or PROFILES / "generated" / args.campaign / profile["cluster_name"]).resolve()
    files = make_plan(args.scheduler, profile, cfg, args.campaign, output,
                      isolate_results=args.isolate_results)
    print("%d %s jobs; ranks/node=%d, threads/rank=%d, cores/rank=%d, cores/node=%d" % (
        len(cfg.nodes), args.scheduler, cfg.ranks_per_node, cfg.threads, cfg.cpus_per_rank,
        cfg.ranks_per_node * cfg.cpus_per_rank))
    print("Results: " + str(cfg.results_dir) + (
        " (separate campaign/cluster/resource directories)" if args.isolate_results
        else " (same paths as run_all.sh; observations append to existing TSVs)"))
    if args.dry_run:
        print(json.dumps(files, indent=2))
        return 0
    # Never change scripts/configuration referenced by an already queued job.
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        text = content if isinstance(content, str) else json.dumps(content, indent=2) + "\n"
        with (output / name).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print("Prepared: " + str(output))
    print("On the cluster, from the repository root (run sbatchman init once):")
    for command in (["sbatchman", "set-cluster-name", profile["cluster_name"]],
                    ["sbatchman", "configure", "--file", str(output / "configs.yaml"), "--overwrite"],
                    ["sbatchman", "launch", "--file", str(output / "jobs.yaml")]):
        print("  " + shlex.join(command))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
