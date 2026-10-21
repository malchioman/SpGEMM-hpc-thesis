# CRESCO tesi-005: 10 campioni e 121 nodi

La campagna ricomincia da zero, con gli stessi eseguibili congelati di tesi-004.
Usa 1, 4, 9, 16, 36, 64, 100 e 121 nodi; 2 processi MPI per nodo; 63 thread
OpenMP e 64 CPU riservate per processo; memoria richiesta di 450000M per nodo.
Ogni variante raccoglie 10 campioni interni, con runs=1, trials=1 e warmup=0.
La validazione principale resta partial. Sono campioni del prodotto con le
strutture predisposte, non dieci lanci MPI indipendenti. I costi iniziali e
full_product_seconds restano misure singole per lancio.

Ci sono 80 job: 10 blocchi per ciascuno degli 8 numeri di nodi. Ogni coppia
di blocchi divide le otto varianti in due gruppi di quattro:

- Gruppo A: le tre baseline e Trident Two-sided.
- Gruppo B: Trident Hybrid, GET, GET Pipeline e PUT.

| Blocchi | Esperimento | Matrici | Prove per job |
|---|---|---|---:|
| 01–02 | Quadrati | mouse_gene, dielFilterV3real, isolates_subgraph4 | 12 |
| 03–04 | Quadrati | cage15, archaea, isolates_subgraph5 | 12 |
| 05–06 | Quadrati | HV15R, eukarya, uniparc | 12 |
| 07–08 | Permutazioni | HV15R, dielFilterV3real, cage15, archaea | 16 |
| 09–10 | Rettangolari | Le dieci matrici, incluso Reddit × restriction | 40 |

Reddit × Reddit resta escluso. Sono 184 prove per numero di nodi, 1472 prove
totali e 14720 campioni principali. I pilot cage8, separati dai risultati
principali, comprendono 24 lanci con verifica completa e 8 con partial e
10 campioni. Ogni blocco verifica i pilot esistenti e completa quelli mancanti;
un errore nei pilot impedisce l'esecuzione delle prove principali di quel job.

## Avvio

Da PowerShell sul PC:

```powershell
scp "C:\Users\Massimo\CLionProjects\SpGEMM-hpc-thesis\scripts\launch_cresco_ten.py" "malchiod@cresco-io.portici.enea.it:/lpor1/store_0/usr/malchiod/tirocinio-spgemm/launch_cresco_ten.py"
```

Da MobaXterm sul cluster:

```bash
python3 /lpor1/store_0/usr/malchiod/tirocinio-spgemm/launch_cresco_ten.py restart
```

`restart` prepara e verifica prima un nuovo snapshot degli eseguibili, sorgenti,
runtime e metadati della campagna precedente; prova le otto richieste Slurm
con `sbatch --test-only`; annulla soltanto i job tesi-004 registrati ancora
PENDING; rifiuta la pulizia se trova job tesi-004 in esecuzione o non riconosciuti.
Quindi elimina definitivamente soltanto:

- `SpGEMM-hpc-thesis/results/tesi-004/partial20-main`
- `SpGEMM-hpc-thesis/test-results/tesi-004/partial20-main`

Non riclona, non ricompila, non scarica matrici. La cache delle matrici resta
esterna e protetta. Sono conservati nel nuovo snapshot la provenienza del
codice e, sotto `retired/`, l'accounting della campagna precedente.

I risultati nuovi sono in
`SpGEMM-hpc-thesis/results/tesi-005/partial10-main/nN/`, suddivisi per esperimento
e variante negli stessi formati TSV. Pilot e diagnostica temporanea sono sotto
`SpGEMM-hpc-thesis/test-results/tesi-005/partial10-main/`.

## Errori, timeout e ripartenza

I blocchi di ciascun numero di nodi usano `afterany`: un blocco successivo può
partire dopo la fine del precedente anche se questo fallisce. Le otto catene
sono indipendenti; la disponibilità e i limiti QoS possono farle attendere.
Non c'è un requisito interno di successo del blocco precedente.

I TSV vengono sostituiti atomicamente dopo ogni prova completa. I lock sono
rilasciati dal sistema operativo alla morte del processo; una directory lock
lasciata dal timeout non blocca la catena. Gli errori delle singole prove vengono
registrati e il runner tenta le altre prove del blocco. Errori nei pilot,
configurazioni incompatibili o dati corrotti interrompono il job.

Il limite rimane 24 ore per job. La divisione riduce il carico per allocazione;
non garantisce una durata massima. Dimezzare i campioni non dimezza lettura,
preparazione, raccolta o validazione seriale.

Il numero di job completati non basta: la completezza si verifica sulle
184 prove con 10 campioni e validazione prevista superata per ogni N.
Ripetere un job controllato salta le sue prove gia complete. `submit` e
`restart` non duplicano invii già registrati e non rilanciano automaticamente
job falliti: questi richiedono un recupero esplicito dopo la diagnosi.
Se un invio viene interrotto prima di registrare l'ID, il launcher si ferma
chiedendo di verificare la ricevuta e la coda prima di qualsiasi nuovo invio.

## Controllo e audit

```bash
python3 /lpor1/store_0/usr/malchiod/tirocinio-spgemm/launch_cresco_ten.py status
python3 /lpor1/store_0/usr/malchiod/tirocinio-spgemm/launch_cresco_ten.py audit
```

`audit` salva l'elenco preciso delle prove mancanti in `audit/missing.json`.
Le analisi derivate vengono generate quando un numero di nodi raggiunge
72 prove quadrate, 32 permutate e 80 rettangolari.

## Verifica locale

`tests/cresco_ten_test.py` usa processi mock per verificare copertura,
campionamento, prosecuzione dopo errori, esclusione dei dati falliti, ripartenze,
scritture atomiche, lock, invii idempotenti e pulizia circoscritta.
Non misura prestazioni, memoria reale o comportamento MPI sul cluster.
