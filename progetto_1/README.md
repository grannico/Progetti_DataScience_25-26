# Progetto 1 — Data Science

Il progetto comprende analisi descrittiva, classificazione e clustering dei giocatori di FC25, oltre al forecasting della serie storica del titolo NVIDIA.

## Struttura

### Analisi descrittiva

- `Analisi_Descrittiva/preparazione_dati.ipynb`: separazione e pulizia di giocatori di movimento e portieri.
- `Analisi_Descrittiva/analisi_descrittiva_finale.ipynb`: statistiche descrittive, distribuzioni, playstyle, età, correlazioni, nazionalità e confronti tra giocatori.
- `Analisi_Descrittiva/mappa_mondo_finale.ipynb`: costruzione e verifica della mappa delle nazionalità.
- `Analisi_Descrittiva/mappa_mondo_finale.html`: versione interattiva della mappa.
- I grafici sono incorporati negli output dei notebook; `figure_eda/` viene generata solo localmente.

### Classificazione

- `Classificazione/classificazione_macroruolo_finale.ipynb`: previsione di attaccante, centrocampista o difensore, confronto tra regressione logistica e Random Forest ed esperimento di feature engineering.
- `Classificazione/classificazione_piede_preferito_finale.ipynb`: previsione del piede preferito con attenzione allo sbilanciamento delle classi.
- Grafici e matrici di confusione sono incorporati nei notebook; `figure_classificazione/` viene generata solo localmente.

### Clustering

- `Clustering/clustering_finale.ipynb`: PCA, K-means, clustering gerarchico e DBSCAN.
- I grafici diagnostici sono incorporati nel notebook; `figure_clustering/` viene generata solo localmente.

### Forecasting

- `Forecasting/forecasting_finale.ipynb`: analisi della serie NVIDIA e confronto tra previsione naïve, ARIMA e SARIMAX.
- I grafici della serie e della diagnostica sono incorporati nel notebook; `figure_forecasting/` viene generata solo localmente.

### Dati

- `data/raw/fc25/`: dataset FC25 originali.
- `data/raw/nvidia/NVDA.csv`: serie storica NVIDIA.
- `data/raw/geografia/world-countries.json`: confini geografici utilizzati dalla mappa.
- `data/puliti/movimento_puliti.csv`: giocatori di movimento dopo la pulizia.
- `data/puliti/portieri_puliti.csv`: portieri dopo la pulizia.

## Ordine consigliato di consultazione

1. Analisi descrittiva e mappa mondiale.
2. Clustering.
3. Classificazione del macroruolo.
4. Classificazione del piede preferito.
5. Forecasting NVIDIA.

I file puliti sono già presenti, quindi non è necessario rieseguire `preparazione_dati.ipynb` per consultare i notebook finali.

## Note metodologiche

- I modelli di classificazione e clustering utilizzano le statistiche tecniche originali e non variabili derivate dalle posizioni dei giocatori.
- Il feature engineering viene valutato separatamente nel notebook del macroruolo, mantenendo invariati modello e suddivisione dei dati.
- DBSCAN è mantenuto come risultato negativo informativo: sul dataset analizzato non produce gruppi sufficientemente compatti e stabili.
- Nel forecasting SARIMAX è incluso tra i modelli del corso, anche se non supera in modo rilevante il riferimento naïve.
- La mappa mondiale rappresenta il 99,59% dei giocatori; le esclusioni residue riguardano piccoli territori assenti dal GeoJSON utilizzato.

Le precedenti versioni di sviluppo sono conservate fuori da questa cartella, in `z_archive/progetto_1_storico`.
