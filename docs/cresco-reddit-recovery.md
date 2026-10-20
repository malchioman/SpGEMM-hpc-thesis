# Recupero della prima ripetizione di tesi-003, a 1 e 4 nodi

I job 5404468 e 5404478 sono falliti su `reddit × reddit`, prima di salvare
`trident_two_sided`. MPI segnala SIGKILL e Slurm riporta circa 492 GiB di MaxRSS
per lo step batch. Il quadro è compatibile con esaurimento della memoria;
il solo SIGKILL non identifica con certezza chi abbia terminato il processo.

La correzione è una compilazione separata, attivata con
`-DSPGEMM_COMPACT_TRIDENT=ON`. L'opzione predefinita rimane OFF. Conserva le righe
dell'accumulatore in vettori compatti, limita le hash table alle righe in
lavorazione, evita una copia globale intermedia durante la raccolta e libera
il prodotto precedente prima di calcolarne un altro. Cambiano anche i costi
di calcolo e raccolta: le misure hanno il suffisso di protocollo `_compact_v1`.

## Cosa viene eseguito

`recover_cresco_missing.sh` prepara due job, per rep-01/n1 e rep-01/n4.
Legge i TSV e le configurazioni congelate, verifica le osservazioni esistenti,
ne copia le righe valide e completa soltanto quelle mancanti. Dai log ricevuti
ci si aspetta 75/192 righe già presenti per ciascun nodo: 9 matrici complete e
le 3 baseline di reddit. Il conteggio effettivo viene calcolato dai file.

- Solo i 5 backend Trident di `reddit × reddit` usano la compilazione compatta.
- Gli altri casi mancanti usano copie degli eseguibili originali, con gli stessi
  2 rank/nodo, 63 thread/rank, 64 CPU/rank e la stessa configurazione di campionamento.
- Ogni job controlla prima la nuova compilazione con 15 prove su cage8:
  5 backend × quadrata, permutata, rettangolare. La verifica numerica è attiva.
  Queste prove restano fuori dalle tabelle prestazionali.
- Le matrici già preparate vengono riutilizzate e verificate tramite hash;
  non vengono scaricate o rigenerate.
- Un errore di un caso viene registrato e il runner prosegue con quelli indipendenti.
  Dopo un errore su reddit compatto evita gli altri tentativi reddit nello stesso job.
  Un errore del pilot interrompe invece il job, perché la correttezza è necessaria.
- Il limite Slurm resta 24 ore. Un arresto dell'intera allocazione può impedire
  di proseguire: il recupero non elimina i limiti di memoria o tempo del cluster.

Le ripetizioni originali a 1 e 4 nodi ancora PENDING vengono messe in hold.
Gli originali già RUNNING continuano; il recupero può partire appena Slurm assegna
altri nodi esclusivi, senza aspettare quegli originali. Temporaneamente possono
esserci sei job attivi invece di quattro. I job a 9 e 16 nodi non vengono modificati. I job messi in hold
restano tali dopo il recupero: non vanno rilasciati indiscriminatamente con il
vecchio eseguibile che ha fallito. Gli ID sono in `metadata/held-original-jobs.json`.

## Avvio sul cluster

Trasferire `reddit-recovery-v1.tar.gz` nella repository del cluster, quindi:

```bash
cd /lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
module load intel/oneapi mpi_flavour/openmpi_intel-4.1.7
source .venv-sbatchman/bin/activate
test ! -e reddit-recovery-v1 && tar -xzf reddit-recovery-v1.tar.gz
bash reddit-recovery-v1/scripts/recover_cresco_missing.sh "$PWD"
```

L'archivio si estrae in una directory separata e la build usa compilatore e flag
della cache CMake originale. Non sovrascrive il codice, i binari o i risultati
della campagna in corso. La compilazione avviene con due processi; le prove MPI
vengono eseguite nei job allocati. Inviare nuovamente lo stesso comando dopo una
submission riuscita non ricompila i binari e non invia job duplicati.

Risultati e manifest sono sotto `results/tesi-003/recovery-reddit-mem01/`.
Per controllare lo stato, da qualunque directory:

```bash
cd /lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
source .venv-sbatchman/bin/activate
python3 reddit-recovery-v1/scripts/recover_cresco_missing.py \
  --manifest results/tesi-003/recovery-reddit-mem01/manifest.json --status
```

L'esito completo è `192/192` per ogni job e `complete: true` in
`rep-01/nN/metadata/completion.json`. Le righe copiate sono elencate in
`metadata/origins.json`; hash dei file originali e dei binari rendono il recupero
tracciabile. Log Slurm e log dei singoli lanci restano disponibili anche in caso
di errore. I file originali non vengono modificati.

Il runner è riprendibile: dopo aver diagnosticato un eventuale nuovo errore e
verificato con `squeue` che il precedente job sia terminato, è possibile inviare
di nuovo il relativo `rep-01/nN/metadata/job.sh` con `sbatch`. Vengono saltate le
osservazioni già presenti. `--submit` da solo è intenzionalmente idempotente e
non reinvia job già registrati. Non rimuovere un lock se il job è ancora attivo.

## Uso dei risultati nella tesi

Le misure compatte di reddit sono una revisione distinta. Per una curva di
scaling reddit coerente bisogna usare la stessa compilazione compatta anche
a 9 e 16 nodi: il runner supporta un piano separato con `--nodes 9 16`, dopo
la fine dei relativi job originali, sostituendo esplicitamente soltanto le cinque
righe Trident reddit. Non mescolare i protocolli nella stessa aggregazione.
Il resto della campagna conserva eseguibili e protocollo originali.

Questo recupero riguarda la prima ripetizione, non tutte e dieci. Eventuali
ripetizioni successive già fallite richiedono un piano distinto selezionato con
`--repetitions`. Non considerare `192/192` della prima ripetizione come il termine
dell'intera campagna.

La verifica locale comprende test MPI dei cinque backend e del trasferimento
a blocchi piccoli, il pilot cage8, la ripresa di una campagna interrotta simulata,
la conservazione dei risultati originali e l'invio senza duplicati. Su un prodotto
sintetico denso 2000×2000, il picco RSS dell'accumulatore è passato da 278084 a
99684 KiB, con identico risultato. Questo test non è reddit e non certifica che
il prodotto completo entri nella memoria di un nodo CRESCO: il lancio sul cluster
è ancora necessario per verificarlo.
