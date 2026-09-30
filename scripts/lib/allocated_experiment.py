#!/usr/bin/env python3
"""Check a scheduler allocation before entering the existing experiment runner."""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import sys

import experiments


def check_allocation(cfg, scheduler, env):
    if len(cfg.nodes) != 1 or cfg.local_check or cfg.hostfile:
        raise ValueError("An allocated job must use one physical node count and no hostfile")
    nodes, ranks, cpus = cfg.nodes[0], cfg.ranks_per_node, cfg.cpus_per_rank
    if scheduler == "slurm":
        if not env.get("SLURM_JOB_ID"):
            raise ValueError("This job must run inside a Slurm allocation")
        allocated = env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES")
        if int(allocated or 0) != nodes:
            raise ValueError("Slurm node count does not match the experiment")
        if int(env.get("SLURM_NTASKS", 0)) != nodes * ranks:
            raise ValueError("Slurm task count does not match nodes * ranks_per_node")
        if int(env.get("SLURM_CPUS_PER_TASK", 0)) < cpus:
            raise ValueError("Slurm allocation has too few CPUs per rank (including Hybrid's worker)")
        placement = env.get("SLURM_TASKS_PER_NODE")
        if placement:
            counts = []
            for group in placement.split(","):
                match = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", group)
                if not match:
                    raise ValueError("Invalid SLURM_TASKS_PER_NODE: " + placement)
                counts.extend([int(match[1])] * int(match[2] or 1))
            if counts != [ranks] * nodes:
                raise ValueError("Slurm ranks per node do not match the experiment")
    else:
        if not env.get("PBS_JOBID") or not env.get("PBS_NODEFILE"):
            raise ValueError("This job must run inside a PBS allocation with PBS_NODEFILE")
        hosts = Counter(Path(env["PBS_NODEFILE"]).read_text(encoding="utf-8").split())
        if len(hosts) != nodes or any(count != ranks for count in hosts.values()):
            raise ValueError("PBS_NODEFILE must contain exactly ranks_per_node entries per physical node; "
                             "check select/mpiprocs and place=scatter")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", required=True, choices=("slurm", "pbs"))
    parser.add_argument("--action", required=True, choices=experiments.RUN_ACTIONS + ("all",))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    runner_args = [args.action, "--config", args.config]
    cfg = experiments.options(runner_args)
    check_allocation(cfg, args.scheduler, os.environ)
    return experiments.main(runner_args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
