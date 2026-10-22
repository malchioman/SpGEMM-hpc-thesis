# Cache del riferimento numerico completo

`--full-cache DIR` riutilizza il riferimento Full senza ricalcolare il prodotto
sequenziale a ogni lancio. La preparazione avviene una volta per coppia ordinata
di input e build compatibile. Ogni lancio confronta comunque il proprio risultato
completo; non vengono salvati o riutilizzati esiti PASS.

Il criterio rimane `componentwise_dot_bound_v1`, descritto in
[numerical-validation.md](numerical-validation.md). Il generatore e il percorso
senza cache condividono il costruttore della riga: stesso ordine dei contributi,
stesso riferimento `long double`, stesso limite di errore. La serializzazione
conserva esattamente i valori, senza convertirli in double o arrotondarli in testo.
Le coordinate cancellate restano nel riferimento. La cache parziale `.spref`
non contiene queste informazioni e non viene convertita in una cache Full.

## Esecuzione

Su un nodo di calcolo, con gli stessi binari congelati dei recuperi:

```bash
prepare_full_reference --matrix-a A.mtx --matrix-b B.mtx --full-cache /scratch/full-references
```

I benchmark usano successivamente `--full-cache /scratch/full-references` e
nessuna opzione `--partial` o `--no-validate`. Il runner Python accetta anche
`"full_cache": "/scratch/full-references"` insieme a `"validate": true`.
I piccoli pilot restano Full senza cache; non richiedono altri riferimenti.

La validazione avviene una volta dopo tutte le ripetizioni, sul risultato finale,
come prima. Kernel, timer scientifici, protocollo, numero di campioni e colonne
TSV restano invariati rispetto alla precedente versione numerica. Nel log:

```text
validation_reference_source=reused
validation_reference_build_seconds=0
```

Sono riportati anche chiave, build, tempo di identità degli input, lettura e
confronto. Il generatore stampa percorso e byte del file completo. Un riferimento
esistente valido viene verificato e riutilizzato anche dalla preparazione.

## Formato e integrità

Il formato binario `SPFR0001` è sequenziale: intestazione, righe e checksum SHA-256
finale. L'intestazione contiene chiave, dimensioni, precisione ed esponente massimo
del riferimento, e dimensione dei record. Ogni riga contiene un conteggio a 64 bit
e coordinate ordinate. Ogni coordinata contiene:

- colonna a 32 bit;
- numero di contributi a 64 bit;
- riferimento storico binary64 per `max_abs_error`;
- riferimento preciso e limite di errore, in rappresentazione binaria esatta.

I long double usano significando intero ed esponente, senza padding ABI. Su x86
con significando a 64 bit ciascun record occupa **44 byte**; sono supportati anche
binary64 e binary128, con identità separata. Il confronto legge una riga alla
volta e verifica il checksum completo prima di poter restituire PASS.

La chiave incorpora hash degli input CSR ordinati (dimensioni, offset, colonne,
valori, duplicati e zeri), build dei sorgenti/riferimento, compilatore e opzioni,
ambiente numerico, precisione e versione del criterio/formato. Input diversi,
permutazioni, ricompilazioni incompatibili o modifiche al criterio richiedono
una preparazione separata. Non si accettano riferimenti di una build diversa.

Cache assente, troncata, corrotta o incompatibile causa errore: **non viene
ricalcolata automaticamente durante il benchmark**. Il generatore non ripara
silenziosamente un file finale corrotto. Non viene scritto un TSV scientifico
quando il caricamento fallisce.

Il generatore usa un lock del sistema operativo e scrive un `.tmp` nello stesso
filesystem, con flush, sincronizzazione e rinomina atomica alla fine. La morte
del processo rilascia il lock; una ripartenza tronca soltanto il proprio `.tmp`
incompleto. Il file del lock non viene cancellato. Locking e sincronizzazione sul
filesystem condiviso di CRESCO vanno verificati sul cluster; i test locali coprono
anche concorrenza e interruzione reale del processo con SIGKILL.

## Spazio e tempo

Per un riferimento con `R` righe e `K` coordinate strutturali, comprese quelle
cancellate, lo spazio con precisione a 64 bit è `176 + 8*R + 44*K` byte.
Con i circa 1,768 miliardi di elementi osservati in HV15R, l'ordine di grandezza è
**78 GB per riferimento, circa 156 GB per originale e permutata**, più eventuali
coordinate cancellate e margine. È una stima, non la dimensione già misurata dei
file completi. Controllare spazio disponibile e quota; il runbook propone scratch.
La memoria temporanea rimane proporzionale alla riga più grande, oltre ad A/B e
al risultato distribuito/raccolto già necessari al benchmark.

Si elimina la ripetizione del calcolo sequenziale e della costruzione dei limiti,
ma restano lettura degli input, hashing, lettura della cache e confronto completo.
Non si promette un tempo preciso o la conclusione entro 24 ore: una preparazione
pesante può ancora richiedere molto tempo, e la lettura condivisa può risentire
di più job concorrenti. Il vantaggio si ammortizza sulle varianti e sui nodi.

## Recupero CRESCO

Il [runbook HV15R](cresco-hv15r-numeric-recovery.md) usa il manifest
`hv15r-roundoff-cached-v4`, una nuova directory e fino a **due job di preparazione**,
uno per il quadrato originale e uno per la permutata seed 42. Se un'azione è già
completa non viene preparato il relativo riferimento. Non sono le 24 preparazioni
parziali della campagna originale.

La sequenza è `prepare`, `submit-references`, attesa di `REFERENCE ... READY`,
`submit-probe`, verifica dei due prodotti completi, `submit-remaining`.
Le preparazioni hanno un proprio budget di 24 ore. I recuperi mantengono 10
campioni e le risorse originali; si inviano soltanto le prove mancanti. Le
ricevute e gli script congelati impediscono invii duplicati e recuperi senza
riferimenti pronti. La campagna originale resta intatta.

## Verifiche

`full_reference` verifica equivalenza di decisioni e diagnostica cached/uncached,
risultati errati anche dopo un PASS precedente, cancellazioni, righe vuote,
precisione long double, subnormali, conteggi a 64 bit, invalidazione, corruzione,
troncamento, dati aggiunti, pubblicazione atomica e lock dopo SIGKILL.

`full_roundoff_hv15r` esegue i casi reali isolati con e senza cache su tutti i
cinque trasporti Trident e sulle tre baseline, confronta i metadati numerici e
conserva l'oracolo razionale esatto. I test offline del recupero verificano
preparazioni separate, riferimenti obbligatori, idempotenza, blocco su errori e
conservazione dei risultati originali. Queste prove non sostituiscono la
validazione dell'intero prodotto HV15R su CRESCO.

Verifica locale del 22 ottobre 2026: **217/217 CTest passati** (436,66 s),
**21/21 test del runner e 10/10 test offline del recupero passati**. La suite
include 15 esecuzioni MPI senza cache e 13 con cache sui casi reali isolati.
Log locale: `test-results/full-cache-ctest.log`.

Su un controllo locale sintetico denso 400 × 400, `a[i,j] = 1 + (17*i + 31*j) % 16`
con indici da zero, Trident Two-sided, due rank locali e un thread/rank,
0 warm-up e 10 campioni, la mediana di tre lanci per modalità è stata:

| Misura | Senza cache | Con cache |
|---|---:|---:|
| Validazione completa | 0,905 s | 0,181 s |
| Durata del lancio | 2,920 s | 2,153 s |

Le modalità sono state alternate, senza eliminare osservazioni. Preparazione:
1,274 s; riferimento: 7.043.376 byte. Sono misure locali WSL su un input sintetico,
non una stima per HV15R/CRESCO o per il filesystem condiviso. La diminuzione della
validazione non equivale alla stessa diminuzione del tempo totale del job.
