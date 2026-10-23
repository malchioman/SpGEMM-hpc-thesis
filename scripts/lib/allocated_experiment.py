"""Check Slurm resources before any benchmark launch."""
import re


def check_allocation(cfg, scheduler, env):
    if len(cfg.nodes) != 1 or cfg.local_check or cfg.hostfile:
        raise ValueError("An allocated job must use one physical node count and no hostfile")
    nodes, ranks, cpus = cfg.nodes[0], cfg.ranks_per_node, cfg.cpus_per_rank
    if scheduler != "slurm":
        raise ValueError("Only Slurm allocations are supported")
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
