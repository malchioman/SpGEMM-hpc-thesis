# Esportazione dei risultati tesi-006

`scripts/export_cresco_results.py` esegue l'audit dell'unione dei dati prima di
creare un archivio scaricabile. Richiede la campagna originale e i job di recupero
terminati, verifica gli snapshot tramite il runner congelato e usa i suoi criteri
di accettazione delle osservazioni. Il piano atteso è di 184 prove per ciascuna
delle otto configurazioni, cioè 1472 osservazioni principali con dieci campioni.
Le due prove preliminari HV15R a quattro nodi sono incluse una sola volta.

```bash
REPO=/lpor1/store_0/usr/malchiod/tirocinio-spgemm/SpGEMM-hpc-thesis
python3 "$REPO/scripts/export_cresco_results.py" \
  --root "$REPO/results/tesi-006/hv15r-roundoff-cached-v4" \
  --output /lpor1/store_0/usr/malchiod/tirocinio-spgemm/tesi006-risultati.tar.gz
```

Il pacchetto mantiene le directory `partial10-cached` e
`hv15r-roundoff-cached-v4`, inclusa `additional-cached/n100`. Conserva TSV grezzi,
array dei campioni, log disponibili, controlli, ricevute Slurm, configurazioni,
manifest e sorgenti congelati. Esclude le directory `metadata/build`,
`metadata/bin` e `references`, oltre alle cache Python e ai file di lock. Le
matrici di lavoro sono esterne a queste directory e non vengono scaricate; i
piccoli input di regressione nei sorgenti rimangono inclusi. Non riscrive né
unisce i TSV, e non rilancia prove. I runner sono conservati come provenienza,
con i percorsi originali del cluster, non come installazione trasferibile.

`export-audit.json` elenca gli esiti per configurazione, osservazioni accettate
e provenienza, prove mancanti, duplicati, casi inattesi e stato di completezza.
Se qualche osservazione manca, l'archivio viene comunque creato e chiaramente
segnalato come incompleto per permettere di scaricare e analizzare i dati
diagnostici. Un job ancora attivo, un accounting incerto o una ricevuta ambigua
bloccano l'esportazione. Un file di archivio già esistente non viene sovrascritto.

Accanto all'archivio viene creato `tesi006-risultati.tar.gz.sha256`. Scaricare
entrambi e confrontare il digest con `Get-FileHash -Algorithm SHA256` prima di
estrarre con `tar -xzf`. Un esempio di cartella locale dedicata è
`C:\Users\Massimo\Downloads\tesi-006-risultati`.

Verifica locale: **5/5 test di esportazione passati**, con la copertura sintetica
1390 originali + 70 recuperi numerici + 12 recuperi parziali, casi mancanti e
duplicati, stato Slurm simulato, contenuto e checksum dell'archivio, protezione
delle sorgenti e integrazione con il runner congelato. I test non attestano la
completezza dei risultati effettivi sul cluster, verificata dal comando sopra.
