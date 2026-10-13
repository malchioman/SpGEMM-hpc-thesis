# Esperimenti della tesi su CRESCO: guida operativa

Preparata il 13 ottobre 2026. L'accesso al cluster non e' ancora disponibile:
nessuna compilazione o misura su CRESCO e' stata eseguita. I blocchi successivi
sono una procedura da completare insieme dopo il primo accesso.

Il repository propone CRESCO8. ENEA documenta Slurm per CRESCO8 nel
[workshop del 25 settembre 2025](https://www.afs.enea.it/project/eneagrid/Seminars/20250925_CRESCO8/workshop_cresco8.pdf).
Partizioni, moduli, account, quote e filesystem vanno verificati nella sessione
reale. La partizione di debug descritta nel workshop permette al massimo due
nodi per utente: non presumere di poterci eseguire il test Trident a quattro nodi.

## 1. Primo accesso: cosa incollare in chat

Segui le istruzioni ricevute con l'account per arrivare al frontend CRESCO
assegnato. Il nodo di ingresso non e' un nodo di calcolo. Sul frontend, in Bash:

```bash
hostname -f
command -v sbatch squeue sinfo scontrol
sinfo -o '%P %a %l %D %c %m'
module -t list 2>&1
module -t avail 2>&1
command -v gcc g++ mpicxx mpirun cmake python3
python3 --version
cmake --version
pwd
df -h .
```

Porta l'output in chat, insieme a eventuali indicazioni ricevute su account e
directory di lavoro condivisa. Non servono password, chiavi o ticket Kerberos.
Da questi dati fisseremo moduli, partizione e percorsi. La scheda della macchina
da citare in tesi verra' raccolta sui nodi assegnati al job, non sul frontend.

## 2. Sorgenti, ambiente e compilazione

Trasferisci il checkout aggiornato, inclusi questa guida e `collect_machine.sh`,
in una directory condivisa visibile dai nodi. Non riutilizzare i binari Windows
o WSL. Congela una revisione identificabile del codice per tutta la campagna;
se cambia il codice o la toolchain, usa una nuova campagna.

Carica i moduli concordati dopo il passo 1: C++ con OpenMP, **Open MPI**,
CMake >= 3.21, Python >= 3.10 per SbatchMan. I nomi dei moduli non sono ancora
noti. Gli script usano opzioni specifiche di Open MPI: un `mpirun` Intel MPI
non e' un sostituto compatibile senza adattamenti. Build e job devono caricare
gli stessi moduli. Dal repository, dopo averli caricati:

```bash
python3 -m venv .venv-sbatchman
source .venv-sbatchman/bin/activate
python -m pip install -r scripts/sbatchman/requirements.txt
sbatchman init

mkdir -p test-results/cresco-setup
set -o pipefail
bash scripts/build.sh --jobs 4 -- -DCMAKE_CXX_COMPILER=mpicxx \
  2>&1 | tee test-results/cresco-setup/build.log

python3 tests/matrix_tools_test.py
python3 tests/experiment_scripts_test.py
python3 tests/sbatchman_scripts_test.py

python3 scripts/matrices.py fetch --ids cage8
python3 scripts/matrices.py prepare --ids cage8 --seed 42
```

`sbatchman init` si esegue una sola volta per checkout. I test Python qui sopra
controllano gli strumenti; il pilot nel passo 4 verifica i programmi MPI reali.
La compilazione a quattro processi e' un punto di partenza da adeguare alle
regole del frontend. Non eseguire l'intera suite MPI sul nodo di ingresso.

## 3. Profilo SbatchMan e risorse

```bash
cp scripts/sbatchman/slurm.example.json scripts/sbatchman/slurm.local.json
```

Compila il profilo: `cluster_name` = `cresco8` se questo e' il cluster assegnato;
`partition`, eventuale `account`, `modules`, eventuale `module_init`, `python`,
`walltime`, `memory_per_node_mb`, `exclusive` = `true`. I 16 GiB e l'ora del
profilo di esempio non dimensionano la campagna finale.

Per il primo pilot usiamo 2 rank per nodo, 4 thread OpenMP e 5 core per rank:
10 core fisici per nodo. Il core aggiuntivo mantiene uguale il budget CPU
anche per `trident_hybrid`, che ha un thread di servizio. Non e' garantito che
quel worker sia fissato separatamente a un core dedicato.

Il layout definitivo si sceglie dopo aver visto socket, NUMA e core dei nodi.
Il default 2 x 4 non va interpretato come utilizzo ottimale di un nodo CRESCO.
Fissiamo lo stesso layout per tutte le varianti e tutti i punti di strong scaling.
Trident richiede **1, 4, 9, 16, ... nodi fisici**; a un nodo non si misura
comunicazione tra nodi. Non usare `--local-check` per misure di tesi.

## 4. Pilot e inventario dei nodi

Genera inizialmente solo il pilot a un nodo:

```bash
python3 scripts/prepare_sbatchman.py --scheduler slurm \
  --profile scripts/sbatchman/slurm.local.json --campaign pilot-001 \
  --action pilot --dry-run -- \
  --nodes 1 --ranks-per-node 2 --threads 4 --cpus-per-rank 5 \
  --results-dir test-results/cresco-pilot-001
```

Controlla l'anteprima e ripeti senza `--dry-run` per creare il piano. Il pilot
esegue tutte le otto varianti sui tre prodotti di cage8, sempre con validazione;
forza un lancio, zero warmup, un repeat e un trial. Non serve a stimare i tempi
delle matrici grandi. Servira' poi un secondo pilot su quattro nodi, con una
partizione che li consenta, per verificare anche i protocolli tra nodi.

**Prima di sottomettere ogni piano**, apri ciascun `.sh` generato sotto
`scripts/sbatchman/generated/<campagna>/<cluster_name>/`. Dopo il caricamento
dei moduli e prima dell'ultima riga `exec`, inserisci:

```bash
# Impostare qui il percorso assoluto della radice risultati DI QUESTO piano.
RESULT_ROOT='/PERCORSO/ASSOLUTO/DEI/RISULTATI'
mpirun -np "$SLURM_JOB_NUM_NODES" --map-by ppr:1:node --bind-to none \
  bash scripts/collect_machine.sh \
  "$RESULT_ROOT/metadata/jobs/$SLURM_JOB_ID" build
```

Per il pilot la radice e' il percorso assoluto di
`test-results/cresco-pilot-001`; per le misure e' la cartella dedicata scelta
al passo 6. Se cambi build directory, cambia anche l'ultimo argomento `build`.
Se il cluster richiede opzioni di trasporto MPI, riporta le stesse anche in
questa chiamata. Il report registra CPU, socket/NUMA, RAM, rete se disponibile,
Slurm job/nodi, software, moduli, CMake, revisione/diff e hash dei binari.
Le sonde mancanti sono segnalate nel report. Non rileva automaticamente il
picco di memoria del benchmark. Non modificare wrapper gia' sottomessi.

Il generatore stampa i tre comandi esatti da eseguire, per esempio:

```bash
sbatchman set-cluster-name cresco8
sbatchman configure --file scripts/sbatchman/generated/pilot-001/cresco8/configs.yaml --overwrite
sbatchman launch --file scripts/sbatchman/generated/pilot-001/cresco8/jobs.yaml
sbatchman status
squeue -u "$USER"
```

Controlla i template in `SbatchMan/configs/cresco8/` prima del lancio. Eventuali
direttive richieste dal sito (per esempio integrazione AUKS) vanno verificate
con il suo esempio di job. Il solo stato COMPLETED non basta: controlliamo
exit code, log e 24 osservazioni PASS per ogni conteggio di nodi del pilot.
Per un fallimento il runner conserva i dettagli nella sottocartella `tmp/`.

## 5. Calibrazione e numero di ripetizioni

Prepariamo prima `cage12`, poi ammettiamo individualmente le matrici della tesi:

```bash
python3 scripts/matrices.py fetch --ids cage12
python3 scripts/matrices.py prepare --ids cage12 --seed 42
```

Prima di ogni campagna grande: un controllo di fattibilita' per prodotto e
configurazione, con `--runs 1 --warmup 0 --repeats 1 --trials 1` e una radice
in `test-results/`. Misuriamo tempo totale del job e memoria; `A*A` puo'
produrre molti piu' nonzero di A. Rank 0 mantiene input globali e raccoglie C
anche con `--no-validate`: aumentare i nodi non elimina questo vincolo.
Se il riferimento seriale entra in memoria e tempo, abilita `--validate`;
altrimenti documentiamo quali casi grandi non hanno quella verifica.

Poi calibrazione temporale con **10 lanci**, `--warmup 3 --repeats 10 --trials 5`,
inizialmente sui casi fattibili e rappresentativi, sempre in `test-results/`.
Controlliamo la sequenza dei campioni, derive e variabilita' tra lanci.
Tre warmup sono una scelta iniziale da verificare; non garantiscono stabilita'.

| Opzione | Proposta iniziale | Significato |
| --- | ---: | --- |
| `--warmup` | 3 | Prodotti esclusi dai campioni, dopo il primo prodotto separato |
| `--repeats` | 10 | Prodotti misurati per trial |
| `--trials` | 5 | Gruppi nello stesso processo MPI |
| `--runs` | 10 | Nuovi lanci MPI per variante/prodotto/configurazione |
| `--campaign-repeats` | 1 | Un solo passaggio dell'intero workflow |

Sono **10 righe TSV con 50 campioni ciascuna**, quindi 500 prodotti misurati,
non 500 repliche indipendenti. Anche lanci distinti nello stesso job possono
condividere disturbi di rete o deriva temporale. Se serve generalizzare tra
allocazioni, distribuiremo le repliche su piu' job, conservandoli separati.

Protocollo statistico proposto, da fissare prima delle misure definitive:

- Metrica primaria: mediana tra lanci del `end_to_end_p90_seconds` di ciascun
  lancio, con intervallo di confidenza al 95%. Descrive il P90 per lancio,
  non il P90 ottenuto unendo tutti i campioni. I dati grezzi restano disponibili.
- Obiettivo di precisione scelto per la tesi: ciascun estremo dell'intervallo
  dista al massimo il 5% dalla stima centrale. Non e' una garanzia universale;
  differenze tra varianti inferiori al 5% possono richiedere piu' precisione.
- Usare la calibrazione separata per fissare il numero finale (inizialmente
  10; aumentabile a 20 o 30, anche oltre se necessario e sostenibile).
  Dieci o trenta lanci non garantiscono automaticamente l'obiettivo.
- Per i CI ricampionare lanci, o blocchi di allocazioni quando c'e' dipendenza;
  non trattare tutti i prodotti interni come osservazioni indipendenti. Con
  pochi lanci i CI sono poco stabili. Dichiarare metodo e limiti, e riportare
  l'incertezza effettiva anche quando supera l'obiettivo. Evitare di fermarsi
  ripetutamente appena un CI nominale risulta stretto senza un disegno sequenziale.
- Il runner mantiene l'ordine delle varianti. Per i confronti finali predisporre
  blocchi con ordini controbilanciati/permutati via `--variants`, salvando
  ordine e configurazione di ogni blocco. Non affermare che gli script lo
  randomizzino automaticamente. Conservare tutti i lanci validi, anche lenti.

Queste sono scelte operative proposte per questa tesi. Il principio di
motivare il campionamento con variabilita' e CI e' discusso da
[Hoefler e Belli, SC15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf).
`full_product_seconds` e' una metrica distinta per distribuzione, setup, primo
prodotto e prima raccolta; si osserva una volta per lancio. Non sommare P90
di fasi diverse. Se un caso non entra su un nodo, lo speedup usera' il piu'
piccolo conteggio fattibile, dichiarandolo.

## 6. Comando della campagna e separazione dei dati

Questo e' un **modello da dimensionare dopo la calibrazione**, non un lancio
da eseguire alla cieca. Mostra cage12 su 1 e 4 nodi con il layout del pilot;
sostituire matrici, layout e numero di lanci con le decisioni finali.

```bash
python3 scripts/prepare_sbatchman.py --scheduler slurm \
  --profile scripts/sbatchman/slurm.local.json --campaign thesis-001 \
  --action all --dry-run -- \
  --nodes 1 4 --ranks-per-node 2 --threads 4 --cpus-per-rank 5 \
  --matrices cage12 --permutation-matrices cage12 \
  --runs 10 --campaign-repeats 1 --warmup 3 --repeats 10 --trials 5 \
  --results-dir results/campaigns/cresco8-thesis-001
```

Togli `--dry-run`, aggiungi la raccolta macchina del passo 4 ai wrapper, poi
esegui i comandi SbatchMan stampati. Attendi il termine prima di sottomettere
un altro piano sulla stessa radice. Un piano crea un job per conteggio di nodi;
ciascun job esegue tutte le sue misure in sequenza. I job sono concatenati con
`afterany`: un fallimento non impedisce necessariamente ai successivi di partire.

Per il piano completo il repository propone 24 prodotti (10 originali,
4 permutati, 10 rettangolari) x 8 varianti x K conteggi di nodi x L lanci.
Con K=2 e L=10 sono **3840 lanci MPI e 192000 prodotti misurati**, piu' pilot,
primi prodotti, warmup, I/O e setup. Walltime e memoria si scelgono sulla
calibrazione; se necessario separiamo matrici e fasi in piu' piani/job.
Non scaricare/preparare `--tier all` prima di verificare spazio e fattibilita'.

La cartella per la chat di analisi sara':

```text
results/campaigns/cresco8-thesis-001/
  strong_scaling/*.tsv
  permutation/*.tsv
  rectangular/*.tsv
  phase_analysis/*.tsv
  matrix_structure/*.tsv
  metadata/
    jobs/<job-id>/<host>.<id>.txt
    plans/
    matrices/
    scheduler/
  pilot/
```

La cartella diventa reale solo dopo l'esecuzione. `--campaign` da solo non
separa i TSV: serve l'esplicito `--results-dir`. I conteggi di nodi restano
nella stessa radice e si distinguono nelle colonne. Per blocchi/allocazioni
separati usare sottocartelle esplicite e conservarle tutte nell'archivio.
Un rilancio accoda righe: non c'e' ripresa automatica o deduplicazione.
Non ricompilare o cambiare gli input mentre sono pendenti job della campagna.

## 7. Chiusura e archivio da consegnare alla chat

A tutti i job terminati, controllare con `sbatchman status` e Slurm che
non ci siano timeout, OOM o errori; contare le righe per variante, prodotto e
configurazione. `sacct` puo' riportare `MaxRSS` solo per alcuni job step:
un campo vuoto non significa memoria zero. Salvare l'accounting dei job reali:

```bash
# Sostituire 12345,12346 con gli ID reali della campagna.
sacct -j 12345,12346 --units=M --parsable2 \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList
```

Dal repository, con la stessa radice del passo 6:

```bash
RESULT_ROOT="$PWD/results/campaigns/cresco8-thesis-001"
mkdir -p "$RESULT_ROOT/metadata/plans" "$RESULT_ROOT/metadata/scheduler"
cp -a scripts/sbatchman/generated/thesis-001 "$RESULT_ROOT/metadata/plans/"
cp scripts/sbatchman/slurm.local.json "$RESULT_ROOT/metadata/"
cp -a matrices/metadata "$RESULT_ROOT/metadata/matrices"
cp test-results/cresco-setup/build.log "$RESULT_ROOT/metadata/"
cp docs/cresco-runbook.md "$RESULT_ROOT/metadata/"
git rev-parse HEAD > "$RESULT_ROOT/metadata/git-commit.txt"
git status --short > "$RESULT_ROOT/metadata/git-status.txt"
git diff --no-ext-diff HEAD > "$RESULT_ROOT/metadata/git-diff.patch"
# Conserva anche i sorgenti effettivi, inclusi eventuali file non ancora in Git.
tar --exclude='scripts/sbatchman/generated' --exclude='__pycache__' \
  -czf "$RESULT_ROOT/metadata/source.tar.gz" CMakeLists.txt src scripts tests
sbatchman status > "$RESULT_ROOT/metadata/scheduler/sbatchman-status.txt"

# Copia pilot e diagnostica: il runner li scrive sotto test-results/.
CHECK_ROOT="$PWD/test-results/campaigns/cresco8-thesis-001"
if [ -d "$CHECK_ROOT/pilot" ]; then cp -a "$CHECK_ROOT/pilot" "$RESULT_ROOT/"; fi
if [ -d "$CHECK_ROOT/tmp" ]; then cp -a "$CHECK_ROOT/tmp" "$RESULT_ROOT/"; fi
```

Salva inoltre l'output `sacct` precedente e i log stdout/stderr dei job in
`metadata/scheduler/`; `scontrol show job ID` indica `StdOut` e `StdErr` mentre
il job e' ancora consultabile. Copia il pilot preliminare e gli ulteriori piani
se usati. Solo dopo questi controlli:

```bash
tar -czf cresco8-thesis-001.tar.gz -C results/campaigns cresco8-thesis-001
sha256sum cresco8-thesis-001.tar.gz
```

Trasferisci **`cresco8-thesis-001.tar.gz`** sul PC e allegalo alla chat di analisi.
Non servono le grandi `.mtx`: i manifest ne documentano identita' e hash.
I risultati vanno copiati anche in uno spazio persistente secondo le regole
del sito; non lasciare l'unica copia in scratch.

Testo da accompagnare all'archivio:

> Analizza questa campagna SpGEMM della tesi. Verifica completezza, protocolli,
> topologia fisica, validazione, configurazioni e provenienza. Usa i TSV di
> strong_scaling, permutation e rectangular come sorgenti: phase_analysis e
> matrix_structure sono esportazioni delle stesse osservazioni, non repliche
> aggiuntive. Distingui lanci MPI, campioni interni e allocazioni; riporta
> l'incertezza, speedup ed efficienza a risorse omogenee. Evidenzia casi mancanti
> o da ripetere senza eliminare arbitrariamente osservazioni lente.
