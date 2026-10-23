# Tooling

| Tool | Purpose |
| --- | --- |
| `campaign_full.py` | Complete campaign with cached numerical validation |
| `campaign_partial.py` | Same campaign with cached row-count validation |
| `campaign.json` | Workloads, resources, sampling and Slurm settings |
| `matrices.py` / `matrices_catalog.json` | Download, inspect and prepare reusable inputs |
| `build.sh` | Release build, with optional CMake arguments |
| `collect_machine.sh` | Per-node machine reports outside measured intervals |

The two campaign entrypoints share `lib/campaign.py`, resource checks and the
measurement parser in `lib/experiments.py`. `experiments.json` supplies internal
runner defaults; edit a copy of `campaign.json` for a new campaign.

See [campaign instructions](../docs/experiments.md) and
[validation](../docs/validation.md). Historical launchers and recovery scripts
are superseded by `submit --retry-failed`. Previously frozen campaigns retain
their own scripts under their result directory and must use those copies.
