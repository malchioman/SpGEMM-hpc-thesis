#!/usr/bin/env bash
# Run on the CRESCO frontend as malchiod. No matrix download or job submission.
set -euo pipefail

requested_parent=/lpor1/store_0/usr/malchiod/tirocinio-spgemm
requested_matrices=/lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/matrices
url=https://github.com/malchioman/SpGEMM-hpc-thesis.git
verified_main=8d726706f71e3339e095b0a5ba8e4a13b7aa5937

fail() { printf 'ERRORE: %s\n' "$*" >&2; exit 1; }
[[ $(id -un) == malchiod ]] || fail "Eseguire sul cluster con l'utente malchiod."
for program in git squeue scancel realpath; do
    command -v "$program" >/dev/null || fail "Comando mancante: $program"
done

cd -- "$requested_parent"
parent=$(pwd -P)
repo="$parent/SpGEMM-hpc-thesis"
[[ -d "$repo/.git" && ! -L "$repo" ]] || fail "Checkout atteso non trovato: $repo"
[[ $(realpath -e -- "$repo") == "$repo" ]] || fail 'Percorso del checkout inatteso.'
[[ -d "$requested_matrices/metadata" ]] || fail 'Cache esterna o metadati delle matrici mancanti.'
matrices=$(realpath -e -- "$requested_matrices")
case "$matrices" in "$repo"|"$repo"/*) fail 'La cache deve essere esterna al checkout.' ;; esac
[[ -s "$matrices/cage8.mtx" ]] || fail 'cage8.mtx non trovato nella cache esterna.'

stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
fresh="$parent/.SpGEMM-hpc-thesis-main-$stamp"
backup="$parent/SpGEMM-hpc-thesis.previous-$stamp"
[[ ! -e "$fresh" && ! -e "$backup" ]] || fail 'Una directory di destinazione esiste gia.'

# Validate the replacement before cancelling jobs or moving the old checkout.
git clone --branch main --single-branch -- "$url" "$fresh"
[[ $(git -C "$fresh" branch --show-current) == main ]] || fail 'Il nuovo checkout non e su main.'
git -C "$fresh" merge-base --is-ancestor "$verified_main" HEAD ||
    fail "main non contiene il commit verificato $verified_main. Vecchio checkout conservato."
git -C "$fresh" grep -q -F 'using CsrOffset = std::int64_t;' -- src/common/csr_matrix.hpp ||
    fail 'Correzione CSR a 64 bit non trovata. Vecchio checkout conservato.'
git -C "$fresh" grep -q -F 'std::vector<CsrOffset> rowPtr;' -- src/common/csr_matrix.hpp ||
    fail 'Offset CSR inattesi. Vecchio checkout conservato.'
git -C "$fresh" grep -q -F 'ValidationMode::Partial' -- src/common/spgemm_common.cpp ||
    fail 'Validazione parziale non trovata. Vecchio checkout conservato.'
git -C "$fresh" grep -q -F '"--partial"' -- scripts/lib/experiments.py ||
    fail 'Opzione --partial non disponibile nel runner. Vecchio checkout conservato.'
new_commit=$(git -C "$fresh" rev-parse HEAD)

# Include the known thesis campaigns and any other job using this checkout.
# Never cancel all jobs of the account indiscriminately.
scoped_jobs() {
    local rows id name workdir
    rows=$(squeue -h -u malchiod -o '%i|%j|%Z') || return
    while IFS='|' read -r id name workdir; do
        [[ -n "$id" ]] || continue
        if [[ "$name" =~ ^tesi00[234]- ]] ||
           [[ "$workdir" == "$repo" || "$workdir" == "$repo/"* ||
              "$workdir" == "$requested_parent/SpGEMM-hpc-thesis" ||
              "$workdir" == "$requested_parent/SpGEMM-hpc-thesis/"* ]]; then
            [[ "$id" =~ ^[0-9]+$ ]] ||
                fail "Job non semplice da controllare manualmente prima del reset: $id"
            printf '%s\n' "$id"
        fi
    done <<< "$rows"
}

selected=$(scoped_jobs)
cancelled=()
if [[ -n "$selected" ]]; then
    mapfile -t cancelled <<< "$selected"
    printf 'Annullo i job del vecchio checkout: %s\n' "${cancelled[*]}"
    scancel "${cancelled[@]}" || fail 'Cancellazione non riuscita; checkout non spostato.'
fi

remaining=$(scoped_jobs)
for ((attempt=0; attempt<20 && ${#remaining}>0; attempt++)); do
    sleep 3
    remaining=$(scoped_jobs)
done
[[ -z "$remaining" ]] ||
    fail "Job ancora presenti dopo 60 secondi: $remaining. Checkout non spostato."

# Keep the old data, configurations and binaries together in one sibling backup.
[[ ! -e "$backup" && ! -L "$backup" ]] || fail 'Destinazione backup gia presente.'
mv -T -- "$repo" "$backup"
if ! mv -T -- "$fresh" "$repo"; then
    mv -T -- "$backup" "$repo"
    fail 'Installazione del clone fallita; vecchio checkout ripristinato.'
fi

mkdir -- "$backup/restart-info-$stamp"
printf 'branch=main\ncommit=%s\nmatrices=%s\ncancelled_jobs=%s\n' \
    "$new_commit" "$matrices" "${cancelled[*]}" > "$backup/restart-info-$stamp/reset.txt"

printf '\nCLONE PULITO PRONTO\nRepository: %s\nBranch: main\nCommit: %s\n' "$repo" "$new_commit"
printf 'Matrici da riutilizzare con --matrix-dir: %s\nBackup precedente: %s\n' "$matrices" "$backup"
printf 'Nessun nuovo job inviato. Il nuovo clone non contiene le vecchie campagne.\n'
git -C "$repo" status --short --branch
