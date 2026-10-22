# HV15R: diagnosi riproducibile degli arrotondamenti

Data: 22 ottobre 2026. Questa indagine locale non modifica gli eseguibili, i
riferimenti o i risultati della campagna CRESCO `tesi-006/partial10-cached`.

## Risultato e limiti

È stato riprodotto con gli eseguibili MPI reali un caso di **HV15R permutata,
seed 42**, in cui il conteggio dei non zeri fallisce esclusivamente per il
diverso ordine delle somme: il sequenziale conserva un residuo minuscolo e
Trident produce zero. Sul caso isolato il controllo parziale passa con un nodo
logico, fallisce con nove, mentre il controllo numerico completo passa per
tutte e cinque le varianti Trident a nove nodi.

È stato inoltre isolato un secondo caso reale in cui la soglia assoluta del
controllo Full (`max_abs_error < 1e-10`) rifiuta Trident, benché il suo risultato
sia più vicino al prodotto esatto degli operandi binary64 rispetto al
riferimento sequenziale. Quindi passare da Partial a Full, mantenendo quella
soglia, **non è una soluzione sufficiente in generale**.

Questi sono controesempi concreti ai criteri di validazione. Non costituiscono
una certificazione dell'intero prodotto HV15R né dimostrano che ogni singolo
fallimento della campagna abbia la stessa causa. Non autorizzano a convertire
i vecchi `PARTIAL_FAIL` in `PASS`.

## Input e metodo

Fonte: [SuiteSparse, Fluorem/HV15R](https://sparse.tamu.edu/Fluorem/HV15R).
Archivio scaricato dal mirror ufficiale UF:
`https://www.cise.ufl.edu/research/sparse/MM/Fluorem/HV15R.tar.gz`.

| File | Byte | SHA-256 |
| --- | ---: | --- |
| `HV15R.tar.gz` | 3455466193 | `e653930ed7b5dbba89fe6d094896a72edf126fcdde6c6d4658b85491190d9b2e` |
| `HV15R.mtx` | 10095089342 | `98e2c2d4e2318a432981a4570861417589ff1346f7e433c324f23f5413acea56` |

I due file sono conservati in `C:/Users/Massimo/Downloads/`. La matrice ha
2.017.169 righe e colonne e 283.073.458 elementi memorizzati. L'hash della
matrice coincide con quello registrato per l'input della campagna.

Per contenere la memoria del PC sono state estratte righe complete di A e tutte
le righe di B da cui dipendono. La permutazione usa la stessa procedura
SplitMix64/Fisher-Yates e lo stesso seed 42 di `scripts/matrices.py`; non è stata
generata sul PC l'intera matrice permutata. Dimensioni globali, indici e ordine
degli elementi all'interno delle righe sono preservati.

Un primo insieme di 6.144 righe non ha mostrato discrepanze di supporto nei
controlli locali. Un secondo insieme di 2.024 righe, selezionate agli indici
originali `127 + 997*k`, ha permesso di trovare il primo caso sotto. Un
programma locale ha individuato i candidati riproducendo l'ordine degli stadi;
la conferma finale è stata eseguita con i veri binari MPI.

Da ciascuna coordinata sono stati poi estratti **tutti** i suoi prodotti scalari
contribuenti. Le coppie minime in `tests/data/hv15r_roundoff/` conservano valori,
indici, dimensioni e ordine rilevante, eliminando le altre coordinate. Sono
input diagnostici sparsi, non matrici per misurazioni prestazionali.

## Caso 1: conteggio diverso, errore numerico minuscolo

Coordinata della matrice permutata, indici a base zero: `(426610, 448836)`.
La riga corrisponde alla riga originale 1119758. I contributi sono dieci.

| Calcolo | Valore |
| --- | ---: |
| Sequenziale binary64 | `-1.8135240096338719e-20` |
| Trident, griglia 3 × 3 | `0` |
| Differenza assoluta | `1.8135240096338719e-20` |
| Prodotto esatto degli operandi binary64, arrotondato per visualizzarlo | `-7.47977200532163e-19` |

Il valore matematicamente esatto **non è zero**: entrambi i risultati binary64
contengono arrotondamenti. Il riferimento sequenziale non è aritmetica esatta.

Risultati effettivi sul caso minimo:

| Nodi logici | Rank MPI | Variante | Controllo | Esito | nnz di C |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 2 | Two-sided | Partial | `PARTIAL_PASS` | 1 |
| 9 | 18 | Two-sided | Partial | `PARTIAL_FAIL` | 0 |
| 9 | 18 | Two-sided, Hybrid, GET, GET Pipeline, PUT (cinque lanci) | Full | `PASS` in tutti | 0 |

Il codice conserva solo valori diversi da zero esatto in
`src/trident/common/matrix_blocks.cpp`. L'ordine degli stadi in
`src/trident/common/execution.cpp` dipende dalla griglia. Con un nodo logico
esiste un solo stadio; con più nodi cambia l'ordine di accumulo dei contributi.
Questo spiega il comportamento del caso isolato ed è coerente con l'assenza
di fallimenti a un nodo riferita dall'utente. Non implica una garanzia universale
di validità a un nodo o di errore a più nodi.

## Caso 2: limite della soglia assoluta Full

Coordinata permutata a base zero `(45207, 1397793)`, 139 contributi,
griglia 4 × 4. Il caso è stato confermato con Trident Two-sided su righe complete
estratte e nuovamente sulla coppia minima, usando 32 rank MPI.

| Quantità | Valore |
| --- | ---: |
| Sequenziale | `73368.519419447824` |
| Trident | `73368.519419447723` |
| Differenza assoluta | `1.0186340659856796e-10` |
| Differenza relativa rispetto al riferimento | circa `1.39e-15` |
| Errore del sequenziale rispetto al prodotto esatto degli operandi binary64 | `6.464144541405697e-11` |
| Errore di Trident rispetto allo stesso prodotto esatto | `3.7221961184510994e-11` |

Il controllo Full restituisce `FAIL` perché confronta con un riferimento
binary64 usando una soglia assoluta fissa. Qui Trident è più vicino al prodotto
esatto. I confronti con l'aritmetica esatta sono stati ottenuti con
`fractions.Fraction` sugli operandi binary64, non con un altro ordine di somme
in double. I valori dell'errore riportati sono arrotondati per visualizzarli.

## Riproduzione dei casi minimi

File conservati in `tests/data/hv15r_roundoff/`:

- `zero-cancellation-witness-{A,B}.mtx` e relativo JSON;
- `absolute-threshold-witness-{A,B}.mtx` e relativo JSON.

I JSON includono tutti gli operandi, l'ordine delle somme e i risultati esatti
riassunti sopra. La build locale era Release (`-O3 -DNDEBUG`), GCC 13.3,
Open MPI 4.1.6, WSL; sorgenti basati su `cc832e28adf6e90ce980cf3e3161cc5d21689445`
con il validatore Full a righe del branch `codex/hv15r-numeric-validation`.
I kernel Trident non sono stati modificati.

Da Linux/WSL nella radice del repository, usando la build diagnostica già creata:

I comandi seguenti documentano la **build diagnostica originaria a soglia assoluta**.
Se la directory è stata ricompilata con `componentwise_dot_bound_v1`, l'ultimo
caso restituisce PASS: la sua vecchia condizione di fallimento è precisamente
uno dei limiti corretti dal nuovo validatore. Usare i test e le istruzioni di
[numerical-validation.md](numerical-validation.md) per la versione aggiornata.

```bash
BUILD="$PWD/test-results/hv15r-validation-build"
DATA="$PWD/tests/data/hv15r_roundoff"
OUT=$(mktemp -d)
export OMP_NUM_THREADS=1
# Impostazioni di compatibilità della prova locale WSL, non della campagna:
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_osc=ucx
export OMPI_MCA_opal_common_ucx_tls=tcp,self
export UCX_TLS=tcp,self

# Atteso PARTIAL_FAIL, exit 1:
mpirun -np 18 --oversubscribe --bind-to none "$BUILD/trident_two_sided" \
  --logical-node-size 2 --threads 1 --warmup 0 --repeats 1 --trials 1 \
  --matrix-a "$DATA/zero-cancellation-witness-A.mtx" \
  --matrix-b "$DATA/zero-cancellation-witness-B.mtx" \
  --partial --results "$OUT/zero-partial.tsv"

# Atteso PASS; ripetibile per tutte e cinque le varianti:
mpirun -np 18 --oversubscribe --bind-to none "$BUILD/trident_two_sided" \
  --logical-node-size 2 --threads 1 --warmup 0 --repeats 1 --trials 1 \
  --matrix-a "$DATA/zero-cancellation-witness-A.mtx" \
  --matrix-b "$DATA/zero-cancellation-witness-B.mtx" \
  --results "$OUT/zero-full.tsv"

# Atteso FAIL con il precedente criterio assoluto, exit 1; PASS col nuovo criterio:
mpirun -np 32 --oversubscribe --bind-to none "$BUILD/trident_two_sided" \
  --logical-node-size 2 --threads 1 --warmup 0 --repeats 1 --trials 1 \
  --matrix-a "$DATA/absolute-threshold-witness-A.mtx" \
  --matrix-b "$DATA/absolute-threshold-witness-B.mtx" \
  --results "$OUT/absolute-full.tsv"
```

Con `-np 2`, il primo comando passa. I nodi sono **logici sullo stesso PC**:
questi lanci verificano l'aritmetica e la validazione, non le prestazioni di rete.
Gli esiti FAIL qui sono quelli attesi e non interrompono una shell ordinaria;
non eseguire la sequenza con `set -e` senza gestirne i codici di uscita.

Log, TSV, estrattori e tracce originali dell'indagine sono conservati localmente
in `test-results/hv15r-local-diagnosis/` (directory ignorata da Git).

## Conseguenze per il recupero

Prima di considerare definitivo il recupero occorre definire e verificare un
criterio numerico motivato, che tenga conto della scala e degli arrotondamenti,
e applicarlo ai prodotti completi. Aumentare una soglia finché questi esempi
passano non sarebbe una giustificazione sufficiente. Non è stata modificata
alcuna tolleranza, alcun kernel o alcuna definizione dei timer in questa indagine.

Successivamente a questa diagnosi è stato implementato sul branch `optimizations`
il [criterio con limite di arrotondamento](numerical-validation.md), con metadati
espliciti e test sui due casi isolati. Il nuovo recupero richiede ancora le prove
complete preliminari sul cluster. La campagna in corso va conservata e i suoi
risultati non vengono riclassificati sulla sola base di questi esempi.
