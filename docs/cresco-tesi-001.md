# Campagna corrente CRESCO: results/tesi-001

Queste indicazioni aggiornano i percorsi e la scelta del campionamento rispetto
ai precedenti esempi della conversazione e del runbook generale.

## Percorsi

- Repository remota: `/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis`.
- Input: `matrices/`, invariato; non interrompere il download in corso.
- Nuova calibrazione: `results/tesi-001/calibrazione-001/`.
- Misure definitive: `results/tesi-001/definitiva-001/rep-XX/nN/`.
- Il runner conserva i controlli pilot separati in `test-results/tesi-001/pilot/`.
  Copiarli nell'archivio finale, senza confonderli con le misure grandi.
- Vecchi pilot e log di download restano in `../risultati/`: non spostarli mentre
  la preparazione e' attiva. Alla fine copiarli sotto `results/tesi-001/metadata/`.

## Campionamento deciso

Configurazione dedicata: `scripts/experiments.cresco-thesis.json`.
Usa l'azione `all` del runner originale, chiamata da `scripts/run_all.sh` e
dall'adattatore `scripts/lib/allocated_experiment.py` usato da SbatchMan.

- 10 ripetizioni dell'intero workflow (`campaign_repeats=10`).
- Un nuovo lancio MPI per caso/variante/configurazione in ogni passaggio (`runs=1`).
- Parametri interni aggiornati: `warmup=0`, `repeats=1`, `trials=1`, `timeout=0`.
- Totale per caso/variante/configurazione: 10 lanci e 10 campioni misurati.
- Ogni valore P90 per lancio coincide con il singolo campione: non va presentato
  come stima stabile del 90esimo percentile. Analizzare la distribuzione dei dieci
  lanci, dichiarando il protocollo senza warmup; non confonderlo con steady-state.
- 10 matrici originali, 4 permutate, 10 prodotti rettangolari, 8 varianti, 1 e 4 nodi:
  3840 lanci principali, piu' 480 piccoli lanci pilot; 4320 lanci MPI complessivi.
- Le analisi delle fasi e della struttura riusano le osservazioni: non sono nuove repliche.

Dieci lanci sono la scelta iniziale per bilanciare costo e stima della variabilita',
non una garanzia di precisione. Riportare la dispersione tra lanci e intervalli di confidenza
al 95%; non garantire a priori un errore del 5%. Anche lanci nella stessa allocazione
possono essere correlati. Eventuali campagne aggiuntive vanno conservate separatamente;
non fermarsi ripetutamente appena un intervallo nominale sembra abbastanza stretto.
Riferimento: https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf

## Piano definitivo concordato il 18 ottobre

Interrompere solo i job della calibrazione 5394420--5394467, conservando gli output.
`scripts/prepare_cresco_run_all.py` prepara venti job che invocano letteralmente
`bash scripts/run_all.sh --config ...`: dieci passate a un nodo e dieci a quattro.
Le dieci ripetizioni sono distribuite dal preparatore: ogni snapshot ha
`campaign_repeats=1`, evitando di moltiplicarle nuovamente per dieci.
Quattro file `lane-1.yaml` ... `lane-4.yaml` hanno ciascuno cinque job sequenziali.
Sottomettere ogni file UNA sola volta con SbatchMan: al massimo quattro job della
campagna possono essere attivi insieme, su nodi esclusivi e risultati separati.
L'ordine delle otto varianti ruota tra ripetizioni; resta documentato negli snapshot.
La rotazione su dieci passate non e' un controbilanciamento perfetto.
Risorse per nodo invariate: due rank, quattro thread, cinque core riservati/rank;
450000 MB e walltime di 24 ore per job. Le quattro sequenze usano al massimo dieci
nodi insieme. La concorrenza puo' comunque introdurre contesa sul filesystem comune.
Il limite di 24 ore resta attivo e la conclusione non e' garantita. Il runner
originale si ferma al primo errore: in tal caso il job rimane incompleto e va
recuperato esplicitamente; la sequenza successiva procede secondo `afterany`.
La calibrazione interrotta non si somma alle dieci repliche definitive.

```bash
python scripts/prepare_cresco_run_all.py \
  --profile ../risultati/pilot-002/metadata/profile.json
```

## Vecchia calibrazione (non rilanciare)

### Estensione della campagna a 9 e 16 nodi

La campagna finale comprende ora 1, 4, 9 e 16 nodi, dieci passate per conteggio.
I venti job a 1/4 nodi (5402333--5402352) restano invariati. Preparare solo
l'estensione in una nuova cartella:

```bash
python scripts/prepare_cresco_run_all.py \
  --profile ../risultati/pilot-002/metadata/profile.json \
  --nodes 9 16 \
  --after-jobs 5402337 5402342 5402347 5402352 \
  --output results/tesi-001/estensione-9-16
```

I quattro ID sono le code delle quattro sequenze gia' inviate, nell'ordine.
Ogni nuova sequenza attende la fine della corrispondente sequenza vecchia.
I nomi delle configurazioni nuove sono distinti: nessun piano esistente cambia.
La dipendenza iniziale afterany e' negli header Slurm; SbatchMan aggiunge i
collegamenti successivi. Massimo quattro job della campagna attivi insieme,
fino a 50 nodi quando lavorano due job da 9 e due da 16, se Slurm li ammette.
Totale finale: 40 job, 7680 osservazioni principali e 960 controlli pilot attesi.
Non risottomettere i piani originali. Conservare entrambe le cartelle e i
corrispondenti output sotto test-results nell'archivio finale.

Il preparatore di calibrazione usa intenzionalmente una sola osservazione per caso:

```bash
python scripts/prepare_cresco_campaign.py \
  --campaign calibrazione-001 \
  --profile ../risultati/pilot-002/metadata/profile.json \
  --output results/tesi-001/calibrazione-001
```

Questo comando prepara 48 job: i comandi SbatchMan stampati li configurano e li
sottomettono. Non eseguire una seconda volta una submission gia' fatta.

La campagna lunga va suddivisa in job compatibili con tempi e memoria osservati
nella calibrazione, mantenendo le 10 ripetizioni complessive. Non copiare il limite
di memoria del pilot (16 GB) e non presumere che tutto entri in un job da 24 ore.
La divisione dei job e l'ordine delle varianti vanno congelati prima delle misure.

Anteprima locale o sul frontend, senza lanciare MPI:

```bash
python scripts/lib/experiments.py all \
  --config scripts/experiments.cresco-thesis.json --dry-run
```

Equivalente attraverso lo script originale (solo dentro allocazioni appropriate
per l'esecuzione reale; sul frontend mantenere `--dry-run`):

```bash
bash scripts/run_all.sh --config scripts/experiments.cresco-thesis.json --dry-run
```

La configurazione nativa esegue tutte le varianti nell'ordine configurato e
accoda ai TSV; non offre ripresa automatica o deduplicazione. Non aggiungere
`--runs 10` a questa configurazione: produrrebbe 100 lanci per caso.
