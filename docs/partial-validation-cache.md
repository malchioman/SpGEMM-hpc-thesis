# Cache persistente della validazione parziale

Per i casi con `PARTIAL_FAIL` dovuto a possibili cancellazioni numeriche, vedere
[il recupero mirato di HV15R con controllo numerico completo](cresco-hv15r-numeric-recovery.md).
Il percorso Full ora usa un riferimento riga per riga e un
[limite numerico di arrotondamento esplicito](numerical-validation.md);
il comportamento Partial descritto qui rimane invariato.

La cache riutilizza il riferimento sequenziale, **mai un esito PASS**. Le otto
implementazioni continuano a verificare dimensioni, finitezza dei valori del loro
risultato e numero di non zeri per riga. `--partial` non confronta tutte le colonne
né i valori numerici: un risultato con valori/colonne errati ma conteggi corretti
può ancora superare questa verifica, esattamente come prima.

## Procedimento e confini delle misure

`buildPartialReference` usa il medesimo `forEachSerialProductRow` usato prima dalla
validazione parziale e ancora usato da `serialSpgemm`: stesso ordine delle somme,
stesso accumulatore sequenziale, esclusione dei soli valori finali esattamente
uguali a zero, controllo di finitezza di tutti i valori finali dell'accumulatore.
Non è un conteggio strutturale. Underflow, cancellazioni esatte e overflow restano
parte del criterio. Il riferimento occupa un vettore di conteggi a 64 bit, senza
costruire il prodotto completo. Anche il percorso senza cache usa questo vettore
per separare costruzione e confronto nei diagnostici.

Audit dei chiamanti:

| Chiamante | Chiusura delle regioni misurate prima di `validateProduct` |
| --- | --- |
| `src/baselines/two_sided_spgemm.cpp` | primo gather e `fullProductSeconds`, warm-up, campioni, gather finale, metriche |
| `src/baselines/one_sided_get_spgemm.cpp` | stesso ordine |
| `src/baselines/one_sided_put_spgemm.cpp` | stesso ordine |
| `src/trident/common/benchmark.cpp` | stesso ordine; chiamante comune a two-sided, hybrid, get, get-pipeline, put |

Le modifiche nei quattro chiamanti aggiungono soltanto il parametro della cache
alla chiamata finale di validazione sul rank 0. Hash degli input, allocazioni del
riferimento, lettura, verifica del checksum e confronto avvengono in quella fase,
dopo i timer. Non sono state aggiunte comunicazioni MPI. Kernel, accumulatori del
prodotto, distribuzione, protocolli, pipeline GET, configurazioni e ordine dei
campioni rimangono invariati. Full e None conservano il loro percorso; la CLI
rifiuta `--partial-cache` senza `--partial` per evitare configurazioni ambigue.

## Identità, formato e invalidazione

La dipendenza aggiunta è **OpenSSL libcrypto**, usata tramite EVP per SHA-256;
servono anche gli header di sviluppo in fase di compilazione. CMake la verifica
con `find_package(OpenSSL REQUIRED COMPONENTS Crypto)`. Se installata in un prefisso
non standard si può usare `OPENSSL_ROOT_DIR` come variabile d'ambiente di CMake.

Il nome è `<chiave-sha256>.spref`. La chiave deriva da:

- contratto esplicito `partial-v1/mm-coordinate-order-duplicates-zeros-preserved/numeric-serial-exact-zero-drop/`;
- fingerprint dei sorgenti del riferimento e del lettore, CMake, compilatore e
  opzioni di compilazione;
- modalità di arrotondamento e, su x86/SSE, bit FTZ/DAZ;
- hash SHA-256 separati di A e B, nell'ordine A × B.

L'hash di ogni matrice comprende dimensioni, lunghezze dei vettori, row pointer,
colonne e rappresentazioni IEEE-754 binary64 dei valori, codificati esplicitamente
in little endian a 64 bit. Include quindi anche l'ordine delle entrate, i duplicati
e gli zeri espliciti: il lettore attuale li conserva, espandendo le simmetrie.
Non ordina né somma preventivamente i duplicati. Il percorso o la data del file
non contribuiscono alla chiave. Un cambiamento effettivo del CSR invalida la cache;
un commento diverso nel file che produce lo stesso CSR non cambia il riferimento.
Il preflight della campagna rifiuta comunque qualsiasi modifica dei byte dei file
rispetto ai manifest originali.

Gli SHA-256 dei file già presenti nel preflight sono riutilizzati per verificare
la provenienza prima della preparazione e dei lanci. Non vengono accettati come
identità dichiarate al binario: non descrivono direttamente ordine e valori del
CSR effettivamente in memoria e non coprirebbero gli input sintetici. Il costo
dell'hash CSR viene misurato separatamente. Lo snapshot congela eseguibili,
sorgenti e configurazioni con i propri SHA-256.

Formato binario v1, lunghezza esatta `160 + 8 * rows` byte:

| Offset | Contenuto |
| --- | --- |
| 0 | 8 byte ASCII `SPPR0001` |
| 8 | 64 byte ASCII della chiave SHA-256 esadecimale |
| 72 | numero righe, unsigned 64 little endian; deve coincidere con A.rows |
| 80 | numero colonne, unsigned 64 little endian; deve coincidere con B.cols |
| 88 | 64 bit little endian: 1 se tutti i valori finali del riferimento sono finiti, altrimenti 0 |
| 96 | `rows` conteggi, ciascuno 64 bit little endian, intervallo `0..INT64_MAX` |
| `96 + 8 * rows` | 64 byte ASCII: SHA-256 di tutti i byte precedenti |

Non ci sono colonne, valori o esiti della validazione di un'implementazione.
La dimensione del file viene controllata prima dell'allocazione del buffer usando
le dimensioni degli input correnti; poi si verificano magic/versione, chiave,
checksum, dimensioni, flag e intervallo dei conteggi. Un riferimento con flag di
finitezza 0 produce sempre `PARTIAL_FAIL`; il preparatore lo salva per diagnosi ma
termina con codice non zero e non pubblica la ricevuta di disponibilità.

Cambiando la procedura o le convenzioni di lettura si deve aggiornare il contratto
o il formato. Il fingerprint automatico causa inoltre un'invalidazione
conservativa al cambiare dei sorgenti rilevanti o della build. Preparatore e otto
eseguibili devono provenire dalla stessa build congelata. Non copiare riferimenti
fra build presumendo che siano compatibili. Il checksum rileva danni accidentali;
la directory deve contenere riferimenti prodotti dal preparatore fidato, non
artefatti numerici costruiti manualmente.

## Cache mancante, concorrenza e interruzioni

La politica è **preparazione separata e lettura obbligatoria**:

- senza `--partial-cache`, il riferimento viene rigenerato a ogni lancio;
- con `--partial-cache DIR`, assenza, corruzione o incompatibilità causano errore,
  codice non zero e nessun PASS. Il benchmark non ricostruisce il riferimento;
- `prepare_partial_reference` è l'unico generatore. Se il file esiste, lo verifica
  e lo riutilizza. Se è danneggiato non lo sovrascrive silenziosamente;
- prima della costruzione il generatore acquisisce un lock esclusivo non bloccante
  `<chiave>.spref.lock` (`flock` su Linux, `LockFileEx` su Windows). Un secondo
  generatore riceve un errore esplicito e può essere riprovato;
- la scrittura usa `<chiave>.spref.tmp` nella stessa directory, flush, fsync,
  chiusura, rename atomico e fsync della directory su Linux. I lettori aprono solo
  il nome finale e non richiedono lock;
- alla morte del processo il kernel rilascia il lock. Il file `.lock` resta e
  **non va eliminato**: cancellarlo durante l'uso creerebbe un secondo inode/lock.
  Il successivo generatore tronca e ricrea l'eventuale `.tmp` interrotto;
- prima del rename un lettore vede assenza/errore; dopo il rename vede l'artefatto
  completo. Un crash dopo il rename lascia un riferimento riutilizzabile.

Usare un filesystem condiviso che supporti lock POSIX fra nodi e rename atomico
nella stessa directory. La verifica locale copre questi meccanismi in Linux;
la semantica del filesystem CRESCO va verificata sui nodi prima della campagna.
Non c'è recupero automatico di un riferimento finale corrotto: isolarlo
manualmente solo dopo aver fermato gli utilizzatori di quella cache e rigenerare
quella coppia. I benchmark non lo accetteranno nel frattempo.

## Preparazione CRESCO8, senza alterare la campagna congelata

`launch_cresco_ten.py prepare` mantiene la funzione precedente: copia i vecchi
eseguibili, che **non acquisiscono queste modifiche**. Non usare `restart` per
questa migrazione: quella funzione preesistente ritira la campagna precedente.
Il nuovo `prepare_cresco_cached.py` non usa ricevute di invio della vecchia
campagna, non annulla job, non cancella risultati e non invia job.

I seguenti comandi sono per l'operatore sul cluster; non sono stati eseguiti da
questa implementazione. Si assume che il checkout `optimizations` contenga tutti
i nuovi file, inclusi quelli ancora non committati, e che tesi-005 sia preparata.

```bash
cd /lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
git branch --show-current
git status --short
export REPO="$PWD"
export SOURCE="$REPO/results/tesi-005/partial10-main"
export CACHED="$REPO/results/tesi-006/partial10-cached"

# Acquisire un nodo di calcolo per snapshot, compilazione e preflight.
salloc --partition=cresco8_cpu --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=450000M --time=24:00:00 --exclusive --hint=nomultithread
srun --nodes=1 --ntasks=1 --cpus-per-task=8 bash -lc '
  source /usr/share/Modules/init/bash
  module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7
  python3 "$REPO/scripts/prepare_cresco_cached.py" prepare \
    --repo "$REPO" --source "$SOURCE" --root "$CACHED"
'
```

La destinazione deve essere nuova. Il comando copia i sorgenti correnti in
`metadata/source`, compila **quella copia** in `metadata/build`, congela i nuovi
binari in `metadata/bin`, aggiorna i runner e configura `partial_cache` per tutti
gli 80 job. Un errore lascia la nuova directory incompleta per ispezione e non
tocca la sorgente: usare un'altra destinazione per il tentativo successivo.
L'ambiente `CXX`/`CXXFLAGS` viene acquisito da CMake nella nuova build; conservare
gli stessi criteri di compilazione scientifica scelti per il cluster. CMake e
libcrypto devono essere disponibili nell'ambiente dei moduli indicati.

Restano 8 configurazioni `[1,4,9,16,36,64,100,121]`, 8 varianti, 23 coppie e
1.472 lanci principali; 10 campioni interni, 0 warm-up, 2 processi/nodo,
63 thread/processo e 64 CPU/processo. Si esclude solo reddit × reddit. Il pilot
cage8 mantiene la validazione completa e il suo campionamento preesistente.
Nessuna variante compatta di Reddit viene introdotta.

La preparazione dei riferimenti ha **24 job distinti**, ciascuno con un nodo,
un processo sequenziale e 24 ore: 23 coppie principali e cage8 quadrato per il
probe parziale aggiuntivo. I tre prodotti del pilot Full non usano la cache.
Questo evita di concentrare tutti i prodotti sequenziali entro un unico limite
di 24 ore. Non è dimostrato che ogni singola coppia termini in quel tempo:
se una coppia supera il limite, non viene resa disponibile e la campagna resta
bloccata. Il calcolo non ha checkpoint per riga; un tentativo interrotto riparte
da capo. Prima di rilanciarlo occorre valutare costo e limite consentito dal centro.

Per eseguire una coppia nell'allocazione corrente, ad esempio la prima:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=1 bash -lc '
  source /usr/share/Modules/init/bash
  module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7
  export OMP_NUM_THREADS=1
  python3 "$CACHED/metadata/runtime/scripts/prepare_cresco_cached.py" reference \
    --root "$CACHED" --pair 0
'
```

I job pronti sono `metadata/reference-jobs/pair-00.sbatch` fino a `pair-23.sbatch`.
La mappa indice/coppia è `reference_pairs` nel manifest. Dopo aver terminato
l'allocazione interattiva, l'operatore può inviarli con al massimo quattro catene
contemporanee; ogni job mantiene il proprio budget di 24 ore:

```bash
declare -a previous=()
i=0
for script in "$CACHED"/metadata/reference-jobs/pair-*.sbatch; do
  lane=$((i % 4))
  dependency=()
  if [[ -n "${previous[$lane]:-}" ]]; then
    dependency=("--dependency=afterany:${previous[$lane]}")
  fi
  job=$(sbatch --parsable "${dependency[@]}" "$script")
  previous[$lane]=${job%%;*}
  printf '%s %s\n' "$script" "${previous[$lane]}"
  i=$((i + 1))
done
```

Non rilanciare l'intero ciclo per una sola coppia fallita: controllare i log
`logs/reference-*.out/.err` e lo stato Slurm, poi riprovare solo il job necessario.
Le ricevute `reference-ready/NN.json` attestano esclusivamente la preparazione
del riferimento, con coppia, chiave e hash del generatore. Nessun PASS di benchmark
vi viene memorizzato. Il launcher verifica che ci siano tutte le ricevute e i file
prima dell'invio e all'avvio dei blocchi, senza caricare i conteggi prima dei timer.

Solo quando la preparazione è completata, l'operatore può usare il normale invio
idempotente degli 80 job, con dipendenze `afterany` fra i blocchi di ogni scala:

```bash
python3 "$CACHED/metadata/launcher.py" submit --root "$CACHED"
python3 "$CACHED/metadata/launcher.py" status --root "$CACHED"
```

`submit` esegue anche il preflight Slurm preesistente. Fallisce prima di chiamare
`sbatch` se manca una ricevuta. I file di riferimento vengono comunque verificati
crittograficamente dal C++ dopo i timer: la ricevuta non sostituisce tale controllo.
Tutti i risultati e i pilot della nuova campagna hanno directory nuove.

## Confronto diretto con e senza cache

All'interno di un'allocazione di un nodo con due task da 64 CPU, dopo avere caricato
i moduli, questo esempio usa cage8 e un eseguibile congelato. Il generatore è
sequenziale e non usa `mpirun`:

```bash
MATRICES=/lpor1/scratch_1/usr/malchiod/tirocinio-spgemm/matrices
BIN="$CACHED/metadata/bin"
REF="$CACHED/references"
mkdir -p "$CACHED/checks"
"$BIN/prepare_partial_reference" --matrix "$MATRICES/cage8.mtx" --partial --partial-cache "$REF"

mpirun --mca btl_vader_single_copy_mechanism none -np 2 \
  --map-by ppr:2:node:PE=64 --bind-to core --nooversubscribe \
  "$BIN/spgemm_two_sided" --matrix "$MATRICES/cage8.mtx" \
  --threads 63 --schedule guided --chunk 64 --warmup 0 --repeats 10 --trials 1 \
  --partial --results "$CACHED/checks/uncached.tsv" \
  > "$CACHED/checks/uncached.log" 2>&1

mpirun --mca btl_vader_single_copy_mechanism none -np 2 \
  --map-by ppr:2:node:PE=64 --bind-to core --nooversubscribe \
  "$BIN/spgemm_two_sided" --matrix "$MATRICES/cage8.mtx" \
  --threads 63 --schedule guided --chunk 64 --warmup 0 --repeats 10 --trials 1 \
  --partial --partial-cache "$REF" --results "$CACHED/checks/cached.tsv" \
  > "$CACHED/checks/cached.log" 2>&1
```

Creare prima `mkdir -p "$CACHED/checks"` se il launcher non ha ancora eseguito i
pilot. Per i runner generici, la stessa opzione è `--partial-cache DIR` oppure
`"partial_cache": "/percorso/assoluto"` nel JSON. Il runner la passa solo quando
il modo effettivo è Partial, quindi non al pilot Full. `null` mantiene il percorso
senza cache. Non modificare i JSON congelati per confrontare i due modi: usare
l'invocazione diretta sopra o configurazioni nuove.

## Diagnostici e verifica

I TSV mantengono schema, nomi e significato dei campi. I log di ogni lancio hanno
una riga con `validation_reference_source` (`regenerated` senza cache, `reused`
con cache), chiave, fingerprint di build e questi tempi `steady_clock` del rank 0:

- `validation_identity_seconds`: hash dei CSR e identità del riferimento;
- `validation_reference_build_seconds`: costruzione numerica sequenziale;
- `validation_cache_read_seconds`: apertura, lettura, checksum e decodifica;
- `validation_compare_seconds`: finitezza e confronto del risultato;
- `validation_cache_write_seconds`: codifica, checksum e pubblicazione, solo preparatore.

Il preparatore riporta inoltre `validation_input_read_seconds` e la sorgente
`generated` oppure `reused`. Non aggiungere questi tempi ai timer scientifici.
Il costo del percorso cached è hashing + lettura/verifica + confronto, non solo
la lettura. Input, metriche, lancio MPI, attese, startup e altre fasi esterne ai timer
possono ancora contribuire al wall time del job. Nessun risultato su CRESCO8 viene
dedotto da quelli locali e non si promettono durate identiche fra esecuzioni.

Test riproducibili su Linux, da eseguire sui nodi di calcolo del cluster:

```bash
# Nuova build standalone, oppure usare metadata/build della nuova campagna.
bash scripts/build.sh --build-dir build-cache --jobs 8
ctest --test-dir build-cache --output-on-failure -j 2
python3 tests/cresco_cached_test.py
python3 tests/cresco_ten_test.py
python3 tests/partial_cache_mpi_test.py --build-dir build-cache --mpi-launcher mpirun -v
# Confronto locale piccolo, 3 lanci per modo, ordine alternato, 10 campioni/lancio:
python3 scripts/compare_partial_cache.py --build-dir build-cache \
  --output test-results/cache-comparison-new
```

`partial_reference_test` confronta il criterio precedente conservato come oracle
con builder, codec e comparatore: prodotti quadrati/rettangolari, righe vuote,
zeri esatti, ordine non associativo, duplicati, underflow/overflow, NaN/Inf,
conteggi errati, forma errata, entrambi gli input modificati, file incompleti,
corrotti/incompatibili, conteggi fino a `INT64_MAX` senza matrici enormi.
`partial_cache_mpi_test.py` esegue 80 lanci reali (8 varianti × 5 casi × 2 percorsi),
con dieci campioni, confronta lo schema TSV e gli esiti, verifica il checksum anche
con `hashlib`, quattro generatori concorrenti e la morte del generatore reale
mentre possiede il lock. `cresco_cached_test.py` verifica isolamento degli snapshot,
24 job, integrazione delle opzioni, ricevute, input modificati e blocco senza riferimenti.

Per le 23 coppie principali, la costruzione può quindi avvenire 23 volte nella
preparazione anziché 1.472 volte nei lanci. Restano tutti i 1.472 confronti dei
risultati. Pilot e probe aggiuntivi restano fuori da quel conto.

## Risultati locali del 21 ottobre 2026

Build Release in Ubuntu/WSL, GCC 13.3, Open MPI 4.1.6 e OpenSSL 3.0.13. La suite CTest completa
ha superato **213/213 test**; dopo gli ultimi controlli di identità della build e
CLI sono stati rieseguiti con successo i tre test mirati (`partial_reference`,
`spgemm_options_smoke`, `partial_cache_mpi`, quest'ultimo include gli 80 lanci MPI).
Non sono misure multi-nodo CRESCO8.

Nei test Python di orchestrazione CRESCO passano 45 test su 47, inclusi i quattro
nuovi e i nove di `launch_cresco_ten.py`. I due errori residui sono in
`cresco_recovery_test.py`: il vecchio recovery si aspetta il protocollo
`trident_staged_csr64_v2_compact_v1`, mentre il mock produce
`trident_staged_csr64_v2`. Gli stessi errori sono stati riprodotti con
`scripts/lib/experiments.py` e `scripts/experiments.json` originali di HEAD in una
copia temporanea. Quel percorso legacy non è stato modificato e la variante
compatta non è stata reintrodotta. I test degli script sono stati eseguiti
isolatamente con `PYTHONPATH=scripts/lib:scripts`; quelli che raccolgono metadati
Git richiedono un checkout Git anche nella copia di prova. I problemi di Bash e
permessi incontrati nel tentativo cumulativo su Windows non riguardano la suite
MPI verificata su Linux.

Il confronto riproducibile `compare_partial_cache.py` ha usato A=B densa 400×400,
due processi MPI locali, un thread ciascuno, zero warm-up e dieci campioni interni.
Tre lanci per ciascun percorso, in ordine alternato; tutti `PARTIAL_PASS`.
Mediane osservate in secondi:

| Costo | Senza cache | Con cache |
| --- | ---: | ---: |
| Costruzione riferimento | 0,961735 | 0 |
| Identità/hash CSR | 0 | 0,007594 |
| Lettura + verifica cache | 0 | 0,005965 |
| Confronto risultato | 0,000093 | 0,000110 |
| Wall time dell'intero lancio MPI | 6,171361 | 5,237136 |

In questo caso locale si osservano circa 0,93 secondi in meno nel lancio completo;
la costruzione era circa 0,96 secondi, dunque **non rappresentava tutto il wall
time**. L'input è un controllo sintetico e la macchina ospitava anche test di
orchestrazione: queste mediane non stimano il risparmio della campagna CRESCO.
Su matrici minuscole dei test, l'inizializzazione di SHA-256 può invece costare
più del prodotto sequenziale. I log e i TSV locali sono conservati in
`build-wsl/cache-comparison-20261021/`, con tutte le osservazioni in
`comparison.json`; il comando sopra consente una ripetizione in una nuova cartella.

La cache Full è distinta da quella dei conteggi: vedere
[full-validation-cache.md](full-validation-cache.md) per preparazione e riuso
del riferimento numerico completo senza modificare il criterio di confronto.
