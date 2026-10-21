#!/usr/bin/env python3
"""Prepare/submit the CRESCO main campaign: 20 internal samples, partial validation.

No matrix downloads, builds, deletions, or algorithm changes. Python 3.8+.
Five dependent blocks per node count reduce each allocation's workload.
"""
import argparse
import copy
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

NODES = (1, 4, 9, 16, 36, 64, 100)
REQUIRED_MAIN = "8d726706f71e3339e095b0a5ba8e4a13b7aa5937"
DEFAULT_REPO = Path("/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis")
DEFAULT_MATRICES = Path("/lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/matrices")
OUTPUT = "results/tesi-004/partial20-main"
MAIN = ("strong_scaling", "permutation", "rectangular")
BLOCKS = (
    ("strong_scaling", ("mouse_gene", "dielFilterV3real", "isolates_subgraph4")),
    ("strong_scaling", ("cage15", "archaea", "isolates_subgraph5")),
    ("strong_scaling", ("HV15R", "eukarya", "uniparc")),
    ("permutation", ("HV15R", "dielFilterV3real", "cage15", "archaea")),
    ("rectangular", ("mouse_gene", "cage15", "HV15R", "dielFilterV3real", "archaea",
                     "eukarya", "isolates_subgraph4", "isolates_subgraph5", "uniparc", "reddit")),
)
RUNTIME = ("scripts/lib/experiments.py", "scripts/lib/allocated_experiment.py",
           "scripts/experiments.json", "scripts/matrices.py", "scripts/matrices_catalog.json",
           "scripts/collect_machine.sh")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(data if isinstance(data, str) else json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify(root):
    manifest = read(root / "manifest.json")
    for name, expected in manifest["sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("File congelato modificato: " + name)
    return manifest


def runner(root):
    runtime = root / "metadata/runtime/scripts/lib"
    sys.path.insert(0, str(runtime))
    import experiments as e
    if Path(e.__file__).resolve() != (runtime / "experiments.py").resolve():
        raise ValueError("Runner importato da un'altra campagna")
    checks = Path(read(root / "manifest.json")["checks"])

    def check_root(path):
        try:
            return checks / path.relative_to(root)
        except ValueError:
            return path

    e.check_root = check_root
    return e


def selected(e, cfg, block):
    action, names = BLOCKS[block - 1]
    return action, [case for case in e.cases(cfg, action) if case[0] in names]


def batch(root, nodes, block):
    q = shlex.quote
    name = f"n{nodes}-b{block}"
    logs = root / "logs" / name
    lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=tesi004-{name}",
             "#SBATCH --partition=cresco8_cpu", f"#SBATCH --nodes={nodes}",
             f"#SBATCH --ntasks={nodes*2}", "#SBATCH --ntasks-per-node=2",
             "#SBATCH --cpus-per-task=64", "#SBATCH --mem=450000M",
             "#SBATCH --time=24:00:00", "#SBATCH --exclusive", "#SBATCH --hint=nomultithread",
             "#SBATCH --output=" + q(str(logs / "%j.out")),
             "#SBATCH --error=" + q(str(logs / "%j.err")),
             "set -euo pipefail", "source /usr/share/Modules/init/bash",
             "module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7",
             "cd -- " + q(str(root / "metadata/runtime")),
             "job_dir=" + q(str(logs)) + '/"${SLURM_JOB_ID}"',
             'mkdir -p "$job_dir"', 'exec > >(tee "$job_dir/launch.log") 2>&1',
             'mpirun --mca btl_vader_single_copy_mechanism none -np "$SLURM_JOB_NUM_NODES" '
             '--map-by ppr:1:node --bind-to none bash scripts/collect_machine.sh "$job_dir" '
             + q(str(root / "metadata/bin")),
             shlex.join(["python3", str(root / "metadata/launcher.py"), "run", "--root", str(root),
                         "--nodes", str(nodes), "--block", str(block)])]
    return "\n".join(lines) + "\n"


def prepare(repo, matrices, root):
    if root.exists():
        manifest = verify(root)
        if len(manifest["jobs"]) != 35 or not manifest["sha256"]:
            raise ValueError("Preparazione precedente incompleta: " + str(root))
        print("Campagna gia preparata: " + str(root))
        return
    if git(repo, "branch", "--show-current") != "main":
        raise ValueError("Preparare dal clone del branch main")
    git(repo, "merge-base", "--is-ancestor", REQUIRED_MAIN, "HEAD")
    git(repo, "diff", "--exit-code", "HEAD", "--")
    commit = git(repo, "rev-parse", "HEAD")
    defaults = read(repo / "scripts/experiments.json")
    for variant in defaults["variants"]:
        if not os.access(repo / "build" / variant, os.X_OK):
            raise ValueError("Eseguibile mancante/non eseguibile: " + variant + "; compilare prima")
    if len(defaults["variants"]) != 8:
        raise ValueError("Attese otto varianti")
    # Snapshot only after the build succeeds. The snapshot survives later checkout edits.
    for name in RUNTIME:
        target = root / "metadata/runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    shutil.copy2(Path(__file__), root / "metadata/launcher.py")
    (root / "metadata/bin").mkdir()
    for name in defaults["variants"] + ["CMakeCache.txt"]:
        shutil.copy2(repo / "build" / name, root / "metadata/bin" / name)
    shutil.copytree(matrices / "metadata", root / "metadata/matrices")
    with (root / "metadata/source.tar").open("wb") as stream:
        subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], stdout=stream, check=True)
    checks = repo / "test-results/tesi-004/partial20-main"
    manifest = dict(commit=commit, checks=str(checks), nodes=list(NODES), jobs=[], sha256={},
                    samples_per_launch=20, independent_launches=1,
                    validation="partial", excluded_pairs=[["reddit.mtx", "reddit.mtx"]])
    write(root / "manifest.json", manifest)
    e = runner(root)
    for n in NODES:
        config = dict(defaults, nodes=[n], ranks_per_node=2, threads=63, cpus_per_rank=64,
                      runs=1, campaign_repeats=1, repeats=20, trials=1, warmup=0, validate="partial",
                      schedule="guided", chunk=64, seed=42, timeout=0, hostfile=None,
                      matrices=["all"], pilot_matrices=["cage8"], permutation_matrices=list(BLOCKS[3][1]),
                      build_dir=str(root / "metadata/bin"), matrix_dir=str(matrices),
                      results_dir=str(root / f"n{n}"), mpi_launcher="mpirun",
                      mpi_args=["--mca", "btl_vader_single_copy_mechanism", "none"])
        write(root / f"metadata/n{n}.json", config)
        cfg = e.options(["all", "--config", str(root / f"metadata/n{n}.json")])
        workload = [selected(e, cfg, b) for b in range(1, 6)]
        keys = [(a, case[2], case[3]) for a, cases in workload for case in cases]
        if len(keys) != 23 or len(set(keys)) != 23:
            raise ValueError("Piano inatteso: devono esserci esattamente 23 coppie distinte")
        if any(a.name == b.name == "reddit.mtx" for _, a, b in keys):
            raise ValueError("Reddit quadrato non ammesso")
        for _, cases in workload + [("pilot", e.cases(cfg, "pilot"))]:
            for _, _, a, b in cases:
                for path in (a, b):
                    if not path.is_file():
                        raise ValueError("Input gia preparato mancante: " + str(path))
        for block, (action, cases) in enumerate(workload, 1):
            key = f"n{n}-b{block}"
            (root / "logs" / key).mkdir(parents=True)
            script = f"metadata/jobs/{key}.sbatch"
            write(root / script, batch(root, n, block))
            manifest["jobs"].append(dict(key=key, nodes=n, block=block, action=action,
                                         expected=len(cases)*8, script=script))
    manifest["sha256"] = {p.relative_to(root).as_posix(): digest(p)
                          for p in (root / "metadata").rglob("*") if p.is_file()
                          and "__pycache__" not in p.parts}
    write(root / "manifest.json", manifest)
    print(f"PREPARATO: {len(manifest['jobs'])} job; 7 configurazioni; 184 prove x 20 campioni per configurazione.")
    print("Dati: " + str(root))


def submit(root):
    manifest = verify(root)
    if not manifest["sha256"] or len(manifest["jobs"]) != 35:
        raise ValueError("Preparazione incompleta; non invio job")
    records = root / "submission"
    records.mkdir(exist_ok=True)
    remaining = []
    for job in manifest["jobs"]:
        receipt = records / (job["key"] + ".json")
        if receipt.exists():
            if not re.fullmatch(r"\d+", str(read(receipt).get("job_id", ""))):
                raise ValueError("Invio precedente incerto; verificare squeue e " + str(receipt))
        else:
            remaining.append(job)
    # Check every resource shape before submitting any job.
    for n in sorted({j["nodes"] for j in remaining}):
        job = next(j for j in remaining if j["nodes"] == n)
        subprocess.run(["sbatch", "--test-only", str(root / job["script"])], check=True)
    for job in remaining:
        args = ["sbatch", "--parsable"]
        if job["block"] > 1:
            previous = f"n{job['nodes']}-b{job['block']-1}"
            complete = root / "state" / (previous + ".complete.json")
            if not complete.exists():
                prior = read(records / (previous + ".json"))["job_id"]
                args.append("--dependency=afterok:" + prior)
        args.append(str(root / job["script"]))
        receipt = records / (job["key"] + ".json")
        # Exclusive creation prevents duplicate submissions, including interrupted invocations.
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(dict(state="submitting", command=args), stream)
        result = subprocess.run(args, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("Risposta sbatch inattesa: " + result.stdout)
        write(receipt, dict(job_id=job_id, command=args, response=result.stdout))
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(job["key"] + " JOB=" + job_id, flush=True)


def existing(e, cfg, action):
    done = set()
    _, warmup, repeats, trials, validation = e.sampling(cfg, action)
    for variant in cfg.variants:
        table = e.output_root(cfg, action) / action / (variant + ".tsv")
        for row in e.read_table(table, e.FIELDS):
            expected = dict(nodes=str(cfg.nodes[0]), ranks=str(2*cfg.nodes[0]), ranks_per_node="2",
                            threads_per_rank="63", cpus_per_rank="64", topology="physical",
                            omp_schedule="guided", omp_chunk="64", warmup=str(warmup),
                            repeats=str(repeats), trials=str(trials), csr_offset_bits="64",
                            benchmark_protocol=e.PROTOCOLS[variant],
                            validation="PARTIAL_PASS" if validation == "partial" else "PASS")
            if any(row[k] != v for k, v in expected.items()):
                raise ValueError("Misura esistente incompatibile: " + str(table))
            e.validate_measurements(row, variant.startswith("trident_"))
            key = (variant, row["matrix_a"], row["matrix_b"])
            if key in done:
                raise ValueError("Misura duplicata: " + str(key))
            done.add(key)
    return done


def perform(e, cfg, action, cases):
    with e.result_lock(e.output_root(cfg, action)):
        done = existing(e, cfg, action)
        remaining = [(case, variant) for case in cases for variant in cfg.variants
                     if (variant, case[2].relative_to(cfg.matrix_dir).as_posix(),
                         case[3].relative_to(cfg.matrix_dir).as_posix()) not in done]
        if remaining:
            headers = e.input_preflight(cfg, [(action, list(dict.fromkeys(c for c, _ in remaining)))])
            for case, variant in remaining:
                one = copy.copy(cfg)
                one.variants = [variant]
                e.run_experiment(one, action, [case], headers)
        final = existing(e, cfg, action)
        expected = {(v, c[2].relative_to(cfg.matrix_dir).as_posix(), c[3].relative_to(cfg.matrix_dir).as_posix())
                    for c in cases for v in cfg.variants}
        if not expected <= final:
            raise ValueError("Blocco incompleto")


def run(root, nodes, block):
    manifest = verify(root)
    if nodes not in NODES or block not in range(1, 6):
        raise ValueError("Configurazione/blocco non ammessi")
    e = runner(root)
    from allocated_experiment import check_allocation
    cfg = e.options(["all", "--config", str(root / f"metadata/n{nodes}.json")])
    check_allocation(cfg, "slurm", os.environ)
    for frozen in (root / "metadata/matrices").glob("*.json"):
        if read(frozen) != read(cfg.matrix_dir / "metadata" / frozen.name):
            raise ValueError("Metadati delle matrici cambiati: " + frozen.name)
    key = f"n{nodes}-b{block}"
    if block > 1 and not (root / "state" / f"n{nodes}-b{block-1}.complete.json").exists():
        raise ValueError("Il blocco precedente non e completo")
    if block == 1:
        pilot = copy.copy(cfg)
        pilot.results_dir = Path(manifest["checks"]) / f"n{nodes}/cage8-full"
        perform(e, pilot, "pilot", e.cases(pilot, "pilot"))
        probe = copy.copy(cfg)
        probe.matrices = ["cage8"]
        probe.results_dir = Path(manifest["checks"]) / f"n{nodes}/cage8-partial20"
        perform(e, probe, "strong_scaling", e.cases(probe, "strong_scaling"))
        print("CAGE8 PASS: 24 verifiche complete e 8 verifiche parziali con 20 campioni", flush=True)
    action, cases = selected(e, cfg, block)
    perform(e, cfg, action, cases)
    if block == 5:
        counts = {a: len(existing(e, cfg, a)) for a in MAIN}
        if counts != dict(strong_scaling=72, permutation=32, rectangular=80):
            raise ValueError("Campagna incompleta: " + str(counts))
        with e.result_lock(cfg.results_dir):
            for analysis in e.ANALYSES:
                e.analyze(cfg, analysis)
        print(f"NODI={nodes}: COMPLETO 184/184, ogni riga contiene 20 campioni", flush=True)
    write(root / "state" / (key + ".complete.json"),
          dict(complete=True, job_id=os.environ["SLURM_JOB_ID"], expected=len(cases)*8))
    print("BLOCCO COMPLETATO: " + key, flush=True)


def status(root):
    manifest = read(root / "manifest.json")
    ids = []
    for n in manifest["nodes"]:
        counts = {}
        for action in MAIN:
            count = 0
            for path in (root / f"n{n}" / action).glob("*.tsv"):
                lines = [line for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]
                count += sum(None not in row and all(v is not None for v in row.values())
                             for row in csv.DictReader(lines, delimiter="\t"))
            counts[action] = count
        completed = sum((root / "state" / f"n{n}-b{b}.complete.json").exists() for b in range(1, 6))
        print(f"NODI={n}: prove {sum(counts.values())}/184 | blocchi {completed}/5 | 20 campioni/prova")
        logs = list((root / "logs").glob(f"n{n}-b*/*/launch.log"))
        if logs:
            lines = max(logs, key=lambda p: p.stat().st_mtime).read_text(errors="replace").splitlines()
            last = next((line for line in reversed(lines)
                         if line.startswith(MAIN + ("pilot:", "CAGE8", "Checking input", "BLOCCO"))), "Avvio")
            print("  " + last)
    for receipt in (root / "submission").glob("*.json"):
        job_id = read(receipt).get("job_id")
        if job_id:
            ids.append(job_id)
    if ids:
        subprocess.run(["sacct", "-X", "-j", ",".join(ids),
                        "--format=JobID,JobName%25,State%22,Elapsed,ExitCode"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "run", "status"))
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--matrices", type=Path, default=DEFAULT_MATRICES)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (args.root or repo / OUTPUT).resolve()
    if args.action == "prepare":
        prepare(repo, args.matrices.resolve(), root)
    elif args.action == "submit":
        submit(root)
    elif args.action == "run":
        run(root, args.nodes, args.block)
    else:
        status(root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERRORE: " + str(error), file=sys.stderr)
        sys.exit(1)
