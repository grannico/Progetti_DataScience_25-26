# Setup del Progetto Chatbot (Rasa)

Questo progetto richiede l'installazione del framework Rasa, l'addestramento del modello e la configurazione di un tunnel ngrok per la comunicazione con Telegram.

## 1. Gestione Ambiente Software
Il progetto utilizza un ambiente Conda per gestire le dipendenze. 

- **Creazione ambiente**: `conda env create -f environment.yml`
- **Attivazione**: `conda activate <nome_ambiente>`

## 2. Configurazione Credenziali e Sicurezza
I file contenenti i token di accesso sono stati esclusi dal tracking di Git.

1. Individuare il file `credentials_template.yml`.
2. Rinominarlo in **`credentials.yml`**.
3. Inserire il proprio **Telegram Access Token** e il **Nome del Bot**.
4. Impostare l'URL del Webhook (vedi sezione successiva).

## 3. Configurazione Webhook (Telegram + ngrok)
Il bot necessita di un indirizzo pubblico HTTPS per ricevere i messaggi.

1. Avviare ngrok sulla porta 5005: `ngrok http 5005`
2. Copiare l'URL generato (es. `https://random-id.ngrok-free.app`).
3. Inserire l'URL nel file `credentials.yml` seguendo questo formato:
   `webhook_url: "https://<tuo_url_ngrok>/webhooks/telegram/webhook"`

## 4. Dataset e File Esterni
Il dataset utilizzato dalle custom actions (`.csv`) è escluso dal repository. Prima di procedere:
- Assicurarsi di inserire il file dataset richiesto all'interno della cartella `actions/`.

## 5. Addestramento ed Esecuzione
Per il corretto funzionamento del bot su Telegram, è necessario seguire rigorosamente questo ordine utilizzando **tre terminali separati**:

### Passaggio 1: Addestramento del modello
Da eseguire nel terminale prima di avviare i servizi:
```bash
rasa train
```

### Passaggio 2: Esecuzione Servizi (Multi-Terminale)
Una volta addestrato il modello, avviare i seguenti servizi in terminali separati:

**Terminale 1: Tunnel ngrok**
```bash
ngrok http 5005
```

**Terminale 2: Rasa Action Server**
```bash
rasa run actions
```

**Terminale 3: Rasa Core** Avviare il bot abilitando le API e i permessi CORS per la comunicazione con Telegram:
```bash
rasa run --enable-api --cors "*"
```