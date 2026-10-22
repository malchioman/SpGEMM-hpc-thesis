# Recupero numerico mirato di HV15R

> **Aggiornamento del 22 ottobre 2026:** il branch `optimizations` implementa ora
> il [criterio numerico con limite di arrotondamento](numerical-validation.md)
> con [cache del riferimento completo](full-validation-cache.md).
> I casi isolati reali sono verificati localmente; i prodotti completi sul cluster
> restano da verificare. Il recupero usa un nuovo manifest `hv15r-roundoff-cached-v4`
> e una nuova directory, senza modificare la campagna congelata.

## Problema osservato e limiti della diagnosi

Nella campagna `tesi-006/partial10-cached`, HV15R × HV15R ha riferimento
sequenziale con 1.768.061.071 elementi. Tutte le cinque varianti Trident producono
1.768.061.070 elementi a 4/9 nodi e 1.768.061.073 a 16 nodi; le tre baseline
superano il controllo parziale. Gli scarti sono **netti**, non il numero di
coordinate o righe discordanti. `PARTIAL_FAIL` non è un timeout né dimostra
da solo un errore numerico significativo. La differenza può derivare dall'ordine
delle somme e dalle cancellazioni, oppure da un difetto condiviso.

Il resoconto successivo dei blocchi conclusi registra 56 prove `PARTIAL_FAIL`:
30 su HV15R originale (cinque varianti, nodi 4/9/16/36/64/100) e 26 su
HV15R permutata con seed 42 (cinque varianti a 4/9/16/36/64 nodi, più
Two-sided a 100 nodi). Non risultano altre matrici in quel resoconto;
la campagna è ancora in corso e l'elenco non è definitivo. Il meccanismo di
arrotondamento è ora confermato su un caso reale della permutata, ma questo
non certifica tutti i prodotti completi falliti.

Un caso sintetico di 8 × 8 (`tests/data/roundoff_stages.mtx`) riproduce il primo
meccanismo con il prodotto reale delle cinque varianti: ordine sequenziale
`1 + 2^-55 - 1` contro ordine a stadi `-1 + 1 + 2^-55`. Il controllo stretto
dei conteggi fallisce; il confronto numerico completo deve accettare il residuo
`2^-55`. Questo test sintetico è ora affiancato dai due casi reali documentati
nell'indagine collegata sopra.

## Modifica e misure

Il percorso Full, selezionato omettendo `--partial` e `--no-validate`, costruisce
un riferimento `long double` una riga alla volta e confronta tutte le coordinate.
Il criterio `componentwise_dot_bound_v1` tiene conto dell'errore del candidato e
del riferimento, usando numero e magnitudine dei contributi e underflow graduale.
Non è una soglia scelta per far passare HV15R. Vedere la derivazione e i limiti in
[numerical-validation.md](numerical-validation.md). NaN, infiniti, CSR malformati
e ambienti numerici incompatibili sono rifiutati.

La memoria aggiuntiva del riferimento dipende dalla riga più grande del prodotto,
non dall'intero C. Il risultato C raccolto, A e B rimangono comunque in memoria.
Il calcolo sequenziale completo e i limiti di errore vengono preparati una volta
per coppia di matrici. Ogni recupero legge il riferimento e confronta il proprio
risultato. La cache richiede spazio consistente (circa 78 GB per riferimento
HV15R su x86, stima); una singola prova per job non garantisce a priori
il completamento entro 24 ore.

Kernel, partizionamento, protocolli MPI, pipeline, accumulatori, definizioni dei
timer, campioni e identificatori `benchmark_protocol` non cambiano. I TSV
aggiungono quattro colonne di metadati: criterio, massimo rapporto errore/limite,
numero di coordinate rifiutate e precisione del riferimento. I vecchi campi
scientifici mantengono significato e unità.
La verifica avviene dopo tutte le regioni misurate. Il log ora riporta strategia,
conteggi delle righe e coordinate discordanti, errore massimo e relativa posizione,
fino a otto esempi (indici a base zero), durata della verifica. Il TSV Full conserva
`validation=PASS/FAIL` e l'errore numerico, distinto dal vecchio `PARTIAL_PASS/FAIL`.
Non si promuovono i vecchi tempi contenuti nei raw TSV falliti.

Partial e la cache restano invariati nel comportamento. La modifica dei sorgenti
invalida conservativamente l'identità di build delle nuove cache: **non ricostruire
la campagna congelata e non sostituire i suoi riferimenti**. Il recupero usa Full
con una nuova cache numerica, in una directory separata.

## Selezione e protezioni del recupero

Nella modalità ordinaria, `scripts/recover_cresco_hv15r.py` richiede tutti gli 80 job originali in stato
terminale, controllato con `sacct`; rifiuta stati attivi, ricevute incomplete o
accounting mancante. Nessuna cancellazione, reset o invio viene effettuato da
`prepare`. Richiede la versione di questo branch sul cluster: i comandi sotto
non aggiornano Git e non presumono che la correzione sia già pubblicata su main.

Il piano seleziona soltanto le prove mancanti delle cinque varianti Trident su
`HV15R.mtx × HV15R.mtx` nello strong scaling e
`permuted/HV15R_permuted_s42.mtx × permuted/HV15R_permuted_s42.mtx` nell'azione
`permutation`. I percorsi e il seed sono ricavati dalla configurazione congelata
originale. Salta separatamente le prove già accettate di ciascuna azione; esclude
baseline, altre matrici permutate, rettangolari e Reddit.
Se mancano anche altri casi, vanno analizzati separatamente; non sono inclusi
implicitamente in questo recupero.

Ogni job misura una sola coppia/variante: 10 campioni, 0 warm-up, 1 trial,
2 rank/nodo, 63 thread/rank, 64 CPU/rank, guided/64, 450000M/nodo,
allocazione esclusiva e 24 ore. Sono possibili al massimo 80 job (due prodotti,
cinque varianti, otto configurazioni di nodi); il numero effettivo dipende dalle
prove mancanti a campagna conclusa. Quelli dello
stesso numero di nodi sono concatenati con `afterany`; gli altri sono indipendenti.
I pilot cage8 della variante sono verificati separatamente prima del caso principale.

La preparazione congela sorgenti, binari, configurazioni e script. Verifica che
il codice di produzione differisca dall'originale soltanto nel validatore Full e
nella serializzazione dei suoi metadati e nel collegamento della cache Full.
Il controllo CMake consente i relativi sorgenti, generatore e test; compilatore, versione
e principali flag devono coincidere con la build originale. Conserva hash dei TSV
originali: modifiche successive ai risultati originali bloccano il recupero.
Invii ripetuti non duplicano i job; una ricevuta ambigua richiede prima di leggere
l'accounting. Un job riuscito non rimisura la prova se rieseguito.

## Preparazione dopo la fine della campagna

Da MobaXterm, nel clone che contiene questa correzione:

```bash
PARENT=/lpor1/store_0/usr/malchiod/tirocinio-spgemm
cd "$PARENT/SpGEMM-hpc-thesis"

sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=hvfull-prepare
#SBATCH --partition=cresco8_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450000M
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/hvfull-prepare-%j.log
#SBATCH --error=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/hvfull-prepare-%j.log
set -euo pipefail
source /usr/share/Modules/init/bash
module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7
REPO=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
cd "$REPO"
python3 scripts/recover_cresco_hv15r.py prepare \
  --repo "$REPO" \
  --source "$REPO/results/tesi-006/partial10-cached" \
  --root "$REPO/results/tesi-006/hv15r-roundoff-cached-v4" \
  --reference-cache-dir /lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/full-validation-references/hv15r-roundoff-cached-v4
SBATCH
```

Se la preparazione fallisce, conservare il log e la directory incompleta per
diagnosi; non sovrascriverla o cancellare i risultati della campagna originale.

## Anticipare riferimenti e prove mentre 100/121 nodi sono ancora in coda

La modalità `prepare --early-probes` permette di anticipare build, riferimenti
e prove preliminari, purché **tutti i dieci job originali a 4 nodi siano conclusi**.
Non basta togliere la dipendenza Slurm alla preparazione ordinaria.
I job della campagna originale continuano con i loro eseguibili congelati.

Se la vecchia preparazione è ancora `PENDING`, dopo aver aggiornato il checkout
si può sostituire soltanto quella con il seguente comando. Usare l'ID effettivo
del job di preparazione da sostituire (nell'esempio `5487188`):

```bash
cd /lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
python3 scripts/start_cresco_hv15r_early.py \
  --replace-prepare-job 5487188 \
  --reference-cache-dir /lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/full-validation-references/hv15r-roundoff-cached-v4
```

Il comando verifica hash e stato dei job a 4 nodi, una destinazione ancora libera,
il nome e il proprietario della preparazione in attesa e la richiesta di risorse.
Poi cancella **solo il job indicato, con filtro `--state=PENDING`**, e invia la
preparazione anticipata senza dipendenze dalla campagna originale. Se la vecchia
preparazione è nel frattempo partita o ha creato la destinazione, si ferma.
La ricevuta in `hvfull-cached-early-control/submission.json` impedisce doppi invii.
Aggiorna `hvfull-cached-prepare.jobid`, quindi i comandi di monitoraggio rimangono:

```bash
PARENT=/lpor1/store_0/usr/malchiod/tirocinio-spgemm
JOB=$(cat "$PARENT/hvfull-cached-prepare.jobid")
sacct -X -j "$JOB" --format=JobID,JobName,State,Elapsed,ExitCode
if [ -f "$PARENT/hvfull-cached-prepare-$JOB.log" ]; then
  tail -n 50 "$PARENT/hvfull-cached-prepare-$JOB.log"
fi
```

La preparazione esegue build e test e invia automaticamente i due riferimenti
numerici. Richiede comunque un'allocazione Slurm e non garantisce l'avvio
immediato. Quando i riferimenti sono `READY`, si usa `submit-probe` come sotto.
Le due prove restano separate a 4 nodi, hanno dieci campioni ciascuna e permettono
di misurare la durata reale prima di decidere come raggruppare i recuperi.

Il manifest preliminare protegge i risultati originali a 4 nodi e tutti i file
congelati. Non legge né congela le tabelle ancora in scrittura degli altri nodi.
Per questi ultimi prepara solo i possibili script, senza inviarli: `run` è
limitato alle due prove e `submit-remaining` rimane bloccato. `status` mostra
esplicitamente la fase preliminare.

**Quando tutti gli 80 job originali saranno conclusi e le due prove saranno
PASS**, aggiornare il piano con:

```bash
RECOVERY=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis/results/tesi-006/hv15r-roundoff-cached-v4
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" finalize --root "$RECOVERY"
```

`finalize` rilegge i risultati ormai stabili, esclude i casi riusciti nel frattempo,
conserva le due prove preliminari, congela gli hash di tutte le tabelle originali
e rende definitivo il piano. Non ricompila, non rigenera riferimenti, non modifica
i vecchi TSV e non invia job. È idempotente. La successiva modalità di invio va
scelta dopo aver valutato i tempi delle prove: `submit-remaining` continua a
inviare un job per combinazione, non un job raggruppato per numero di nodi.

## Preparazione dei due riferimenti condivisi

Dopo `Prepared ... Nothing submitted.`, verificare spazio disponibile e quota su
scratch: prevedere circa 156 GB complessivi per i due riferimenti, più margine.
La stima dipende dal supporto strutturale effettivo, incluse le cancellazioni.

```bash
df -h /lpor1/scratch_1/usr/malchiod/tirocinio-spgemm
RECOVERY=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis/results/tesi-006/hv15r-roundoff-cached-v4
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" submit-references --root "$RECOVERY"
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" status --root "$RECOVERY"
```

Sono al massimo due job indipendenti da un nodo, un processo, 450000M e 24 ore:
uno per originale, uno per permutata. Non ricostruiscono i 24 riferimenti parziali.
Aspettare `REFERENCE strong_scaling: READY` e `REFERENCE permutation: READY`
(per le azioni presenti). Un errore blocca i recuperi, senza avviare implicitamente
il calcolo sequenziale in ogni benchmark. I log sono `logs/reference-*.log`.

## Prove preliminari, poi eventuale recupero restante

Dopo che i riferimenti richiesti sono pronti:

```bash
RECOVERY=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis/results/tesi-006/hv15r-roundoff-cached-v4
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" submit-probe --root "$RECOVERY"
```

Il comando `submit-probe` invia una prova per ciascuna azione che ha casi mancanti:
preferisce `n4-trident_two_sided` per l'originale e
`n4-permutation-trident_two_sided` per la permutata. Se una di queste è già
accettata nella campagna originale, seleziona la prima prova mancante della
stessa azione. Se un'azione è già completa, non prepara alcuna prova per essa.
Le due prove sullo stesso numero di nodi sono concatenate con `afterany`.
Hanno gli stessi 10 campioni del recupero e, quando superano il controllo
completo, quei campioni vengono conservati: non vengono ripetute.

```bash
RECOVERY=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis/results/tesi-006/hv15r-roundoff-cached-v4
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" status --root "$RECOVERY"
```

Se tutte le prove preliminari superano il controllo numerico, il comando seguente
invia le altre coppie selezionate. È bloccato finché ogni prova iniziale non ha
un TSV PASS coerente e una ricevuta di completamento. Se una fallisce, leggere
`checks/n4/tmp/trident_two_sided-*/launch.log` e il raw TSV: servono errore massimo
ed esempi delle coordinate, senza allargare automaticamente la tolleranza.

```bash
RECOVERY=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis/results/tesi-006/hv15r-roundoff-cached-v4
python3 "$RECOVERY/metadata/source/scripts/recover_cresco_hv15r.py" submit-remaining --root "$RECOVERY"
```

## Dati per l'analisi

I nuovi TSV sono in `hv15r-roundoff-cached-v4/nN/strong_scaling/trident_*.tsv` e
`hv15r-roundoff-cached-v4/nN/permutation/trident_*.tsv`. Gli originali
rimangono in `partial10-cached/nN/...`; lo script non aggiunge righe ai vecchi TSV.
Il vecchio monitor richiede `PARTIAL_PASS` e rifiuterebbe righe Full mescolate
senza aggiornare la sua logica. All'esportazione finale si uniranno le tabelle
su `(azione, matrix_a, matrix_b, nodi, variante)`, conservando provenienza,
modalità di validazione e manifest delle due build, senza doppioni.

La modalità numerica più forte riguarda solo i recuperi; non rende retroattivamente
numericamente validate le altre osservazioni parziali. La verifica reale di HV15R
su CRESCO resta necessaria prima di dichiarare risolto il caso osservato.

## Verifica locale del 22 ottobre 2026

Le verifiche del nuovo criterio sono descritte in
[numerical-validation.md](numerical-validation.md). Sono distinti dai precedenti
215 test del validatore a soglia assoluta: l'accettazione è ora basata sul limite
per coordinata. I test MPI includono tutti i cinque trasporti Trident e i casi
reali isolati, anche quello che superava la vecchia soglia `1e-10`.

Esito locale della versione aggiornata: **217/217 test CTest e 10/10 test offline
del recupero passati**. Questi esiti non attestano l'installazione sul cluster
né la validazione numerica dell'intero prodotto HV15R su CRESCO.

I test offline `python tests/cresco_hv15r_test.py` verificano selezione delle sole
prove mancanti, campioni, idempotenza, blocco su campagna attiva o prove preliminari
fallite, dipendenze e conservazione dei TSV originali. Non dimostrano prestazioni
o correttezza numerica dell'intero HV15R sul cluster.

L'estensione per le prove anticipate supera **17/17 test offline del recupero**
e **6/6 del comando di sostituzione della preparazione**, oltre ai 21 test del
runner. Sono verificati: altri nodi ancora attivi, immutabilità dei dati a 4 nodi,
blocco degli invii prematuri, selezione finale aggiornata, conservazione dei
campioni preliminari, assenza di ricompilazione durante `finalize`, doppio invio,
ID/nome/proprietario errati e preparazione che parte durante la sostituzione.
I comandi Slurm di questi test sono simulati; nessun job reale è stato inviato.
