# Validazione numerica con limite di arrotondamento

Il percorso Full usa `streaming_roundoff_v2`, criterio
`componentwise_dot_bound_v1`. Confronta tutte le coordinate del prodotto,
una riga alla volta, con un riferimento `long double` e un limite deterministico
per gli arrotondamenti. Non modifica i kernel, i protocolli MPI o i timer.
Con `--full-cache` il riferimento e i limiti sono riutilizzati senza ricalcolo,
con lo stesso criterio: vedere [full-validation-cache.md](full-validation-cache.md).
Partial, la sua cache dei conteggi e `--no-validate` conservano il comportamento.

L'implementazione è nel blocco `full-streaming-validation-v2` di
`src/common/spgemm_common.cpp`. Le altre modifiche C++ serializzano i metadati
della validazione; lo script di recupero verifica questa separazione rispetto
ai sorgenti congelati della campagna.

## Criterio e derivazione

Per una coordinata `(i,j)`, siano `m` il numero di contributi scalari effettivi,
`c = sum_k a_ik b_kj` il prodotto esatto degli operandi binary64, e
`S = sum_k |a_ik b_kj|`. Si contano anche contributi nulli e duplicati presenti
negli input. Il conteggio è a 64 bit e gli overflow vengono rifiutati.

Il modello assume base 2, arrotondamento al più vicino e underflow graduale.
Sono esclusi overflow intermedi del riferimento, NaN, infiniti, fast-math e
modalità di arrotondamento diverse. Il controllo verifica ambiente numerico e
CSR prima del confronto; su x86 verifica separatamente anche FTZ e DAZ.

Con al più `m` moltiplicazioni e `m` addizioni per coordinata, si usa la stima
conservativa, per una precisione con unit roundoff `u` e minimo subnormale `eta`:

```
t = 2*m
gamma = t*u / (1 - t*u)
D = t*eta / (1 - t*u)
|computed - c| <= gamma*S + D
```

Il termine `D` usa `eta` per operazione, più conservativo di `eta/2`, per coprire
l'underflow graduale. Le cinque varianti Trident accumulano ogni coordinata nello
stesso accumulatore durante gli stadi; i trasporti spostano gli operandi senza
introdurre riduzioni numeriche aggiuntive. Anche i kernel delle baseline rientrano
in questo modello. Una moltiplicazione-addizione fusa richiede meno arrotondamenti
e resta coperta. Un futuro kernel con operazioni differenti richiede una nuova
verifica del modello.

Il riferimento `r` e la somma delle magnitudini `s` sono calcolati in
`long double` (`L`). Anche questi calcoli hanno un errore, che non viene ignorato:

```
S_upper = (s + D_L) / (1 - gamma_L)
bound = (gamma_double + gamma_L)*S_upper + D_double + D_L
accept coordinate iff upper(|actual - r|) <= bound
```

La stima di `S_upper` segue da `|s - S| <= gamma_L*S + D_L`.
Il limite tiene conto quindi sia del prodotto candidato sia del riferimento.
È una specializzazione conservativa delle stime classiche per somme e prodotti
scalari; si veda [Rump, Error Estimation of Floating-Point Summation and Dot Product](https://www.tuhh.de/ti3/paper/rump/altRu11.pdf),
in particolare il modello con underflow e le stime basate sulla somma dei valori
assoluti. Non vengono usate ipotesi probabilistiche o soglie tarate su HV15R.

I calcoli del limite avvengono in `long double` con arrotondamento conservativo
tramite `nextafter`: numeratori verso l'alto, denominatori verso il basso.
Il confronto usa una maggiorazione della differenza. Se `2*m*u >= 0.5`, il
limite non è finito oppure il riferimento non è disponibile, la coordinata
viene rifiutata. Non si considera un limite infinito come autorizzazione al PASS.

La precisione effettiva è registrata: su GCC/x86-64 verificato localmente il
significando `long double` ha 64 bit, contro i 53 di `double`. Su piattaforme
dove coincidono, il criterio include comunque l'errore del riferimento ma
non beneficia di precisione aggiuntiva.

## Coordinate, struttura e significato del PASS

Il confronto copre l'unione delle coordinate del prodotto di riferimento e del
risultato. Un elemento non memorizzato equivale a zero; una coordinata generata
ma cancellata non viene omessa dal confronto. Una colonna spuria senza alcun
contributo ha limite zero. CSR malformati, colonne fuori intervallo o duplicate
nel risultato e valori non finiti sono rifiutati.

PASS significa che ogni errore è compatibile con il limite dichiarato, **non**
uguaglianza bit a bit o prova formale dell'implementazione. Il limite è di caso
peggiore: in forte cancellazione può accettare differenze relative grandi
rispetto a un risultato quasi nullo. Non garantisce cifre significative relative
al valore finale né esclude errori algoritmici più piccoli del limite. I casi
critici possono essere approfonditi con aritmetica esatta, come nei due
[casi reali di HV15R](hv15r-roundoff-diagnosis.md).

La validazione corrente controlla il risultato raccolto dopo le ripetizioni,
come prima; non ricontrolla separatamente ogni campione cronometrato. Non rende
retroattivamente validi i precedenti `PARTIAL_FAIL` o numericamente completi i
`PARTIAL_PASS` della campagna congelata.

## Metriche e compatibilità dei dati

Il campo storico `max_abs_error` conserva la distanza massima dal sequenziale
binary64, ma **non decide più da solo il PASS Full**. La funzione legacy
`validationPassed(double)` resta disponibile con soglia `1e-10` per i vecchi
controlli che la usano esplicitamente. Il percorso Full applica il nuovo criterio.
Un riferimento sequenziale binary64 non finito viene rifiutato conservativamente,
anche se il riferimento più preciso fosse finito.

Quattro colonne di metadati sono aggiunte ai TSV, senza cambiare campioni,
intervalli temporali, unità, P90 o `benchmark_protocol`:

| Campo | Full | Partial / None |
| --- | --- | --- |
| `validation_criterion` | `componentwise_dot_bound_v1` | `row_nnz_exact_v1` / `none` |
| `validation_max_error_ratio` | massimo errore maggiorato / limite | `NA` |
| `validation_rejected_coordinates` | coordinate rifiutate | `NA` |
| `validation_reference_digits` | bit del significando del riferimento | `NA` |

Un PASS Full richiede zero coordinate rifiutate, rapporto finito tra 0 e 1 e
metadati completi. I nuovi script accettano `max_abs_error > 1e-10` soltanto
quando i metadati identificano questo criterio e sono coerenti. I file storici
privi dei campi conservano il criterio `legacy_absolute_v1` e la vecchia soglia;
non vengono riclassificati. L'analisi preserva i nuovi campi negli export.
L'append diretto con intestazione incompatibile è vietato: usare una nuova
directory. Il recupero usa tabelle separate e scritture atomiche.

I log riportano inoltre numero di righe controllate, differenze di supporto,
massimo errore rispetto al riferimento preciso, peggior rapporto errore/limite,
coordinata, valori e primi otto esempi. Non si eliminano contributi nei kernel
per far coincidere artificialmente i conteggi.

## Verifiche locali e passaggio al cluster

- `spgemm_full_streaming`: prodotti esatti di controllo, cancellazioni, scale
  piccolissime, underflow, input/risultati malformati, errori in valori e colonne,
  FTZ/DAZ e rounding mode; entrambi i casi reali e loro versioni corrotte.
- `full_streaming_mpi`: cinque varianti, quattro nodi logici, dieci campioni;
  Partial rifiuta la cancellazione sintetica, Full la accetta.
- `full_roundoff_hv15r`: 15 lanci MPI sui due casi reali (1/9/16 nodi logici),
  tutte le varianti Trident e le tre baseline; verifica indipendente degli
  operandi con `fractions.Fraction` e controlli dei metadati da parte del parser.
- `cresco_hv15r_test.py`: orchestration offline, protezioni e recupero mirato;
  nessun job reale viene inviato.

Compilare in Release e usare `ctest --output-on-failure`. La compatibilità MPI
locale WSL può richiedere `OMPI_MCA_osc=ucx`,
`OMPI_MCA_opal_common_ucx_tls=tcp,self`, `UCX_TLS=tcp,self`; questi override non
vengono inseriti nei job CRESCO.

Esito della verifica locale del 22 ottobre 2026: **217/217 test CTest passati**
in una suite completa con GCC 13.3/Open MPI 4.1.6 in WSL, inclusi i confronti
con e senza cache numerica completa. I **10 test offline di
recupero** passano su una copia temporanea dei sorgenti in WSL; comprendono
anche il rifiuto di righe legacy presentate come nuovi risultati e l'accettazione
di un errore assoluto superiore a `1e-10` soltanto con i metadati appropriati.
È inoltre verificato il rifiuto di un rapporto appena superiore a 1 anche quando
una conversione a binary64 lo arrotonderebbe a 1.

Il log CTest completo è conservato localmente in `test-results/full-cache-ctest.log`.
Le misurazioni di durata dei test locali non sono stime dei tempi di CRESCO.

La memoria del riferimento è proporzionale alla riga più grande; A, B e il
risultato raccolto rimangono residenti. Senza `--full-cache` rimane il costo del ricalcolo completo; con la cache
restano lettura e confronto, mentre il calcolo è spostato nelle preparazioni: non c'è garanzia di
completare un prodotto HV15R entro 24 ore. Prima del recupero generale servono
le prove complete preliminari sul cluster, separatamente per originale e
permutata, come nel [runbook di recupero](cresco-hv15r-numeric-recovery.md).
