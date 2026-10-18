# CSR a 64 bit e trasferimenti grandi

Le otto varianti usano `CsrOffset = std::int64_t` per puntatori CSR, conteggi di
elementi e offset nei payload. Dimensioni e indici di riga/colonna rimangono `int`
a 32 bit. Somme dei prefissi, conversioni delle dimensioni, allocazioni e dimensioni
delle finestre RMA hanno controlli espliciti; il lettore Matrix Market accetta
conteggi di elementi a 64 bit e conserva duplicati e zeri come prima.

`src/common/mpi_large.hpp` trasmette `CsrOffset` con `MPI_INT64_T`, indici con
`MPI_INT` e valori con `MPI_DOUBLE`. Send/Recv, Isend/Irecv, GET e PUT suddividono
il payload in blocchi con conteggio rappresentabile da `int`. Questo usa API
MPI 3.1, disponibili in Open MPI 4.1.x, senza dipendere dalle routine MPI-4 `_c`.
Dimensioni delle finestre e displacement sono verificati contro `MPI_Aint`;
le finestre dei puntatori CSR hanno displacement unit `sizeof(CsrOffset)`.
Completamento, flush e sincronizzazioni dei backend restano nei punti originari.

Il limite di ogni blocco è `SPGEMM_MPI_CHUNK_LIMIT` (default 2147483647 elementi).
Tutti i processi di un job devono usare gli eseguibili della stessa build e lo
stesso limite, perché mittente e destinatario devono suddividere i messaggi
allo stesso modo. Non mescolare eseguibili precedenti con quelli CSR64.

## Ricompilazione sul cluster

Dopo aver sincronizzato sorgenti, CMake, test e runner aggiornati, nell'ambiente
con il modulo Open MPI della campagna:

```bash
bash scripts/build.sh --build-dir build-csr64 --jobs 8 -- \
  -DSPGEMM_MPI_CHUNK_LIMIT=2147483647
```

Nelle invocazioni del runner/SbatchMan selezionare `--build-dir build-csr64` e
una directory nuova, ad esempio `--results-dir results/tesi-csr64`.
Mantenere allocazione Slurm, nodi, rank, thread e campionamento della campagna.
Eseguire prima il pilot con validazione; solo successivamente rilanciare i casi
grandi. Le misure CSR32 già riuscite restano dati della versione precedente.

Il campo `benchmark_protocol` identifica ora la versione CSR64 per ciascuna
variante; `csr_offset_bits` e `mpi_chunk_limit` vengono conservati nei TSV grezzi
e normalizzati. Il runner aggiornato rifiuta eseguibili con il vecchio protocollo
e non aggiunge righe a tabelle con un header incompatibile.

## Verifiche riproducibili

```bash
ctest --test-dir build-csr64 --output-on-failure

bash scripts/build.sh --build-dir build-csr64-test --jobs 8 -- \
  -DSPGEMM_MPI_CHUNK_LIMIT=7 \
  '-DCMAKE_CXX_FLAGS=-Wconversion -fsanitize=undefined -fno-sanitize-recover=all'
ctest --test-dir build-csr64-test --output-on-failure
```

Su localhost possono servire le normali opzioni Open MPI di oversubscription.
Il limite 7 serve ai test, non alla campagna di prestazioni. La suite include:

- `csr_large_counts`: prefissi sopra INT_MAX, somma globale di blocchi locali,
  pianificazione di messaggi oltre INT_MAX, rifiuto di overflow a 64 bit e di
  allocazioni/finestre non rappresentabili, senza allocazioni enormi.
- `csr_large_transfer`: Send/Recv, Isend/Irecv, GET e PUT per int, int64 e double,
  conteggi vuoti e ai bordi dei blocchi; metadati con valori sopra INT_MAX;
  displacement non nullo e sentinelle attorno alla destinazione RMA.
- Le verifiche numeriche, di lifecycle, pipeline, richieste hybrid e contratto
  delle misure esistenti, eseguite con tutte le otto varianti.

La prova con blocchi piccoli verifica i percorsi di segmentazione. Una prova
con buffer realmente oltre INT_MAX elementi e il prodotto completo di eukarya
richiedono ancora risorse del cluster; non sono sostituite dai test piccoli.

Verifica locale del 18 ottobre 2026: tutti i 179 test CTest superati nella build
Release e tutti i 179 nella build con limite 7 e UBSan; superati anche i 5 test
Python degli script della campagna CRESCO. Ambiente: WSL, GCC 13.3, Open MPI
4.1.6. Nei primi passaggi 51 test per build fallivano con `MPI_ERR_WIN` usando
`btl_vader_single_copy_mechanism=none`; il problema è stato riprodotto anche con
l'eseguibile CSR32 precedente. Le 51 verifiche sono state ripetute con successo
senza quell'opzione locale. La configurazione CRESCO non è stata cambiata.
Il test aritmetico è stato inoltre ricompilato e rieseguito in entrambe le build
dopo l'aggiunta del caso sul limite INT64_MAX.

## Raccolta globale e memoria

La raccolta completa su rank 0 resta attiva anche con `--no-validate`; restano
invariate le fasi incluse nei tempi. La rappresentazione ora supera INT_MAX
nonzero, ma deve comunque stare in memoria. Il solo CSR finale richiede circa
`12 * nnz(C) + 8 * (righe(C) + 1)` byte, oltre a input, copie, hash table e altri
temporanei. La raccolta Trident mantiene anche i temporanei per riga. Questa
modifica elimina il limite dei conteggi a 32 bit, non il limite della RAM.
