# Chatbot di ricette con Rasa

Il progetto implementa un chatbot che suggerisce ricette a partire dagli ingredienti disponibili e può guidare l'utente durante la preparazione. L'interfaccia principale è Telegram, collegata a Rasa tramite un tunnel HTTPS di ngrok.

Il motore combina un punteggio sugli ingredienti con la similarità TF-IDF. Gli ingredienti sono confrontati come elementi strutturati e normalizzati, non come semplici parole isolate. Le 298 ricette hanno inoltre una tipologia assegnata esplicitamente, usata insieme ai filtri per difficoltà e tempo. Le richieste indicate come “veloci” o “rapide” usano una soglia di 30 minuti.

La ricerca dà priorità alle ricette che contengono tutti gli ingredienti indicati. Per categorie generiche come carne, pesce o formaggio, il bot chiede prima una precisazione. Se non esiste una corrispondenza completa, risponde in modo colloquiale e propone possibili alternative invece di mostrare risultati parziali senza spiegazioni. Dopo una ricerca può suggerire un ingrediente aggiuntivo; se l'utente lo rifiuta, l'ingrediente viene escluso e la ricerca viene aggiornata.

Sono supportati anche:

- ingredienti esclusi, per esempio `senza formaggio`;
- modalità rigorosa, flessibile e libera;
- preferenze vegetariana, vegana, senza glutine e senza lattosio;
- ricerca di una ricetta per titolo, anche quando l'entità NLU non viene estratta;
- avvio testuale della preparazione e scelta della ricetta se ne sono state mostrate più di una;
- proposte casuali compatibili con ingredienti, esclusioni e preferenze già impostati.

I filtri alimentari derivano dagli ingredienti presenti nel dataset e non sostituiscono una verifica medica di allergie o contaminazioni.

## 1. Ambiente software

Su Windows:

```bash
conda env create -f environment_win.yml
conda activate chef_bot
python -m pip install -r requirements.txt
```

Su macOS:

```bash
conda env create -f environment_mac.yml
conda activate chef_bot
```

## 2. Credenziali

I token reali non sono salvati nel repository.

1. Copiare `credentials_template.yml` come `credentials.yml`.
2. Inserire il token Telegram e il nome del bot.
3. Impostare l'URL del webhook come descritto nella sezione successiva.

## 3. Telegram e ngrok

1. Avviare `ngrok http 5005`.
2. Copiare l'URL HTTPS generato.
3. Impostare in `credentials.yml`:

```yaml
webhook_url: "https://<url-ngrok>/webhooks/telegram/webhook"
```

## 4. Dataset e metadati

Il dataset CSV usato dalle custom action è escluso dal repository e deve essere inserito in `actions/dataset_ricette_giallozafferano.csv`.

Se cambia soltanto l'ordine delle righe del dataset, riallineare i metadati e rigenerare la lookup:

```bash
python scripts/sync_recipe_metadata.py
```

La classificazione è associata ai titoli, non agli indici delle righe. Lo script rifiuta titoli mancanti, nuovi o duplicati invece di assegnare categorie errate. Se vengono aggiunte nuove ricette, occorre prima classificarle manualmente in `actions/tipologie_ricette.json` e poi eseguire lo script.

## 5. Addestramento e avvio

Addestrare prima il modello:

```bash
rasa train
```

Avviare poi tre terminali separati.

Terminale 1, tunnel pubblico:

```bash
ngrok http 5005
```

Terminale 2, custom action:

```bash
rasa run actions
```

Terminale 3, server Rasa:

```bash
rasa run --enable-api --cors "*"
```

## 6. Verifica

Controllo dei dati Rasa:

```bash
rasa data validate
```

Test unitari del motore e delle custom action:

```bash
python -m unittest discover -s tests -p "test_recipe_engine.py" -v
```

Test dei percorsi conversazionali:

```bash
rasa test core --stories tests/test_stories.yml --out results/core --no-plot
```

Cross-validation NLU a tre fold, includendo anche la lookup dei titoli:

```bash
rasa test nlu --nlu data --cross-validation --folds 3 --out results/nlu --no-plot
```

Valutazione manuale su formulazioni non presenti nei dati di addestramento:

```bash
rasa test nlu --model models/<modello>.tar.gz --nlu tests/nlu_test.yml --out results/nlu_holdout --no-plot
```

I report JSON prodotti da Rasa vengono salvati nelle sottocartelle di `results/`.
Il test manuale integra la cross-validation ma non la sostituisce; per la relazione vanno riportate separatamente entrambe le misure e il metodo con cui sono state ottenute.
