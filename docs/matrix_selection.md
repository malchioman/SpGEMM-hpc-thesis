# Selezione delle matrici: verifica delle fonti

Verifica del **8 ottobre 2026**. Proposta: mantenere come candidati principali
le dieci matrici della Tabella 2 del paper Trident GPU. Tutte hanno almeno una
fonte che ha risposto alla lettura parziale del file durante questa verifica.
**L'identità esatta con gli input usati dagli autori non è ancora verificata.**
La corrispondenza di nomi, dimensioni e conteggi è un riscontro di identificazione,
non una prova di uguaglianza di coordinate, valori e preprocessing.
Non emerge quindi, per ora, la necessità di sostituirle per irreperibilità.
La selezione finale per gli esperimenti CPU resta subordinata ai pilot di memoria
e tempo su CRESCO-8.

Riferimenti locali: [catalogo](../scripts/matrices_catalog.json),
[piano degli esperimenti](experiments.md), [downloader](../scripts/matrices.py).
Questa nota documenta la verifica e non modifica il catalogo o la campagna.

## Risultato dei controlli

- I quattro URL su `sparse-files.engr.tamu.edu` hanno fallito con una connessione
  interrotta (`WinError 10054`) da questa macchina. Questo non dimostra che i file
  siano stati rimossi o che il server fallisca da ogni rete.
- I quattro corrispondenti URL su `www.cise.ufl.edu` hanno risposto con HTTP 206
  e dati gzip. Sono il mirror già previsto dal downloader del progetto.
- Tutti i sei URL diretti NERSC hanno risposto con HTTP 206 e intestazioni
  Matrix Market coerenti con dimensioni e conteggi del catalogo.
- Sono stati letti al massimo 64 KiB per richiesta. Per tre archivi SuiteSparse
  è stato inoltre possibile leggere l'intestazione della matrice decomprimendo
  il prefisso. Per `HV15R` il prefisso non basta a raggiungere l'intestazione
  della matrice nell'archivio: le dimensioni restano quelle della scheda
  SuiteSparse e del paper.
- Non sono stati scaricati integralmente i dataset, calcolati hash completi o
  verificati tutti gli elementi. Un prefisso valido non garantisce che una
  successiva trasmissione completa termini correttamente.

## Fonti candidate verificate per raggiungibilità

Nella colonna `nnz paper` sono riportati i numeri della Tabella 2, senza
uniformare artificialmente le diverse convenzioni di memorizzazione.
Le dimensioni dei download provengono da `Content-Range`; GB = 10^9 byte.
Gli archivi `.tar.gz` sono compressi, i file `.mtx` NERSC no.

| Matrice | Righe = colonne | nnz paper | Download verificato | GB trasferiti per download completo |
| --- | ---: | ---: | --- | ---: |
| HV15R | 2.017.169 | 283.073.458 | [Florida, tar.gz](https://www.cise.ufl.edu/research/sparse/MM/Fluorem/HV15R.tar.gz) | 3,455 |
| mouse_gene | 45.101 | 28.967.291 | [Florida, tar.gz](https://www.cise.ufl.edu/research/sparse/MM/Belcastro/mouse_gene.tar.gz) | 0,164 |
| archaea | 1.644.227 | 204.792.654 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/HipMCL/archaea/arch_vs_arch_30_50length_propermm.mtx) | 4,402 |
| eukarya | 3.243.106 | 359.763.936 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/HipMCL/eukarya/euk_vs_euk_30_50length_propermm.mtx) | 7,977 |
| isolates_subgraph4 | 4.372.771 | 264.799.194 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/subgraph4_iso_vs_iso_30_70length_ALL.m100.oneindexed.mtx) | 5,918 |
| isolates_subgraph5 | 2.186.385 | 66.399.522 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/subgraph5_iso_vs_iso_30_70length_ALL.m100.oneindexed.mtx) | 1,450 |
| cage15 | 5.154.859 | 99.199.551 | [Florida, tar.gz](https://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage15.tar.gz) | 0,418 |
| uniparc | 2.856.197 | 29.621.573 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/Incremental-HipMCL/uniparc_active_p1_vs_uniparc_active_p1_id50_cov80.mtx) | 0,639 |
| reddit | 232.965 | 57.307.946 | [NERSC, mtx](https://portal.nersc.gov/project/m1982/GNN/reddit.mtx) | 0,918 |
| dielFilterV3real | 1.102.824 | 89.306.020 | [Florida, tar.gz](https://www.cise.ufl.edu/research/sparse/MM/Dziekonski/dielFilterV3real.tar.gz) | 0,600 |

Queste dimensioni descrivono il trasferimento, non lo spazio complessivo per
estrazione, originali, permutazioni, restrittori e risultati, né la RAM necessaria.

## Identità dei dati e simmetria

Le intestazioni lette direttamente sono:

| Matrice | Storage dichiarato | Entry memorizzate nell'intestazione |
| --- | --- | ---: |
| mouse_gene | real symmetric | 14.506.196 |
| cage15 | real general | 99.199.551 |
| dielFilterV3real | real symmetric | 45.204.422 |
| archaea | real general | 204.792.654 |
| eukarya | real general | 359.763.936 |
| isolates_subgraph4 | real symmetric | 264.799.194 |
| isolates_subgraph5 | real symmetric | 66.399.522 |
| uniparc | real general | 29.621.573 |
| reddit | real symmetric | 57.307.946 |

Per `mouse_gene` e `dielFilterV3real`, il catalogo e SuiteSparse riportano nnz
della matrice espansa. Per `isolates_subgraph4`, `isolates_subgraph5` e `reddit`,
il numero del paper coincide invece con le entry memorizzate nel file simmetrico.
Il lettore CPU attuale espande gli elementi fuori diagonale e conserva le
coordinate duplicate; i kernel ne accumulano i contributi nel prodotto.
Il numero di righe del file non va assunto come nnz finale in CSR.

Nei prefissi dei due file `isolates` si osservano coordinate in entrambi i
triangoli nonostante il flag `symmetric`. Questo richiede una verifica completa
delle coppie e dei duplicati prima di rivendicare una riproduzione numerica
dell'input GPU. Nomi, dimensioni e conteggi uguali non provano uguaglianza di
tutti i valori o del trattamento applicato dal codice originale. Non cambiare
silenziosamente il flag e non sostituire versioni con conteggi diversi come se
fossero mirror identici. Il piano del progetto già documenta questa distinzione.

### Verifica aggiuntiva del repository degli autori

Lo [script di lancio a c37debac](https://github.com/HicrestLaboratory/Trident/blob/c37debaccc58b72859f1837f260900a40094848c/scripts/make_scripts.py)
imposta `MAT_DIR` a
`/global/cfs/cdirs/m4646/hns_spgemm_matrices_pico/known_squaring_nnz/`
e costruisce per Trident percorsi `<gruppo>/<nome>/<nome>.bmtx`.
Nei manifest e script esaminati non è presente una corrispondenza verificabile
con checksum tra queste copie locali e gli URL pubblici qui elencati.
La revisione esaminata è quella di riferimento del progetto CPU; non è stata
certificata come l'esatta revisione di ogni esecuzione riportata nel paper.

Il [lettore della revisione collegata](https://github.com/HicrestLaboratory/distributed_mmio/blob/2e7c9b3c3205f4f019269b155f9381c8c4974b43/src/mmio/io.cpp)
espande ogni elemento fuori diagonale quando il file dichiara `symmetric`.
Il [convertitore MTX/BMTX](https://github.com/HicrestLaboratory/distributed_mmio/blob/2e7c9b3c3205f4f019269b155f9381c8c4974b43/src/tools/mtx_to_bmtx.cpp)
legge la matrice attraverso quel lettore; il percorso di scrittura binario
usa l'intestazione `general` dopo l'espansione. Questo spiega una trasformazione
possibile, ma non dimostra che gli input degli esperimenti siano stati preparati
proprio con questa versione e queste opzioni.

Stato dell'identificazione:

- `HV15R`, `mouse_gene`, `cage15`, `dielFilterV3real`: identificazione solida
  tramite le voci ufficiali SuiteSparse e i dati del paper; uguaglianza esatta
  con le copie usate nelle esecuzioni GPU non certificata.
- `archaea`, `eukarya`, `uniparc`: file NERSC compatibili con nomi, dimensioni
  e conteggi del paper; manca una prova diretta del collegamento alle copie GPU.
- `isolates_subgraph4`, `isolates_subgraph5`, `reddit`: stessa compatibilità
  dei metadati, con un'ulteriore ambiguità fra conteggi memorizzati ed espansi
  e, per isolates, coordinate in entrambi i triangoli.

Per chiudere la verifica occorrono gli input effettivi degli autori, oppure
un manifest affidabile con provenienza, hash e passaggi di conversione.
Un hash calcolato solo sui nostri download consente la riproducibilità futura,
ma non prova da solo l'identità con dati degli autori non disponibili.
Per confrontare MTX e BMTX serve il confronto dei dati decodificati, tenendo
conto anche della precisione numerica, non l'uguaglianza binaria dei file.

## Criterio proposto per la selezione definitiva

1. **Reperibilità:** conservare per ora tutte e dieci le matrici del paper;
   il mirror Florida risolve il problema osservato per SuiteSparse.
2. **Correttezza del flusso:** usare `cage8` come pilot, già presente localmente;
   `cage12` e `web-Google`, già nel catalogo, restano casi ausiliari. Non sono
   sostituti identici delle matrici del paper e non bastano da soli per una
   campagna di scaling rappresentativa.
3. **Fattibilità CPU:** ammettere ciascun prodotto `A*A` e `A*R` dopo un pilot.
   Registrare nnz effettivo, distribuzione degli nnz per riga, nnz di C, tempo
   e picco di memoria. La dimensione del file e il solo nnz di A non bastano:
   contano le moltiplicazioni candidate e il riempimento di C.
4. **Limiti dell'implementazione:** gli indici e i conteggi CSR sono `int`;
   anche il risultato deve rientrare nel limite implementato. Rank 0 conserva
   input globali e raccoglie C; `--no-validate` non elimina questi costi.
5. **Eventuali sostituzioni successive:** sceglierle per famiglia applicativa,
   dimensioni, gradi delle righe, struttura/località e costo del prodotto,
   motivando la proprietà che si conserva e quella che cambia. Un grafo web
   di dimensioni simili non diventa per questo equivalente a una rete proteica.

Per il confronto originale/permutato, mantenere la coppia con lo stesso input
e lo stesso seed. Non trasferire al caso CPU i valori di imbalance della
Tabella 2: sono riferiti alla distribuzione GPU a 256 GPU.

## Fonti

- PDF fornito dall'utente, *Communication-Avoiding SpGEMM via Trident Partitioning
  on Hierarchical GPU Interconnects*, sezioni 5.2-5.4, Tabella 2 a pagina 9.
- Schede SuiteSparse: [HV15R](https://sparse.tamu.edu/Fluorem/HV15R),
  [mouse_gene](https://sparse.tamu.edu/Belcastro/mouse_gene),
  [cage15](https://sparse.tamu.edu/vanHeukelum/cage15),
  [dielFilterV3real](https://sparse.tamu.edu/Dziekonski/dielFilterV3real).
- Directory NERSC: [archaea](https://portal.nersc.gov/project/m1982/HipMCL/archaea/),
  [eukarya](https://portal.nersc.gov/project/m1982/HipMCL/eukarya/),
  [subgraphs](https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/).
  Per `uniparc` e `reddit`, la verifica utile è stata la richiesta diretta al
  file riportato in tabella, anche quando la consultazione web della directory
  non ha restituito contenuti.
- Controlli HTTP parziali eseguiti in questa sessione sui file elencati.
