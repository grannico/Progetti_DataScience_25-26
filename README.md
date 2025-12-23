# Progetti Data Science 2025/2026

Questo repository raccoglie i progetti e le analisi sviluppate per il corso di Data Science durante l'anno accademico 2025/2026.

L'obiettivo è coprire l'intero ciclo di vita del dato: dall'analisi esplorativa e mining, passando per l'analisi di reti complesse, fino alle moderne applicazioni di NLP e Generative AI.

## 📂 Indice dei Progetti

Di seguito una sintesi dei moduli che compongono il progetto d'esame. Per dettagli specifici, documentazione e codice, fare riferimento al **README** all'interno di ciascuna sottocartella.

| # | Progetto | Descrizione | Stack Tecnologico | Cartella |
| :--- | :--- | :--- | :--- | :---: |
| **01** | **Analisi Dati & Mining** | Pipeline completa di Data Science: Analisi descrittiva (EDA), Classificazione, Clustering e Forecasting su Serie Temporali. | *Pandas, Scikit-learn, Seaborn, Matplotlib* | [Vai](./progetto_1) |
| **02** | **SNA: Game of Thrones** | Social Network Analysis applicata ai personaggi e alle relazioni nella serie TV *Il Trono di Spade*. | *NetworkX, Gephi, Pandas* | [Vai](./progetto_2) |
| 03 | Chatbot FrigoChef | Assistente virtuale basato su Rasa: ricerca ricette tramite algoritmo semantico (TF-IDF) e guida interattiva passo-passo. | Rasa, Python, Scikit-learn, Telegram | [Vai](./progetto_3) |
| **04** | **Sentiment Analysis** | Analisi del sentimento applicata a dataset testuali (recensioni/commenti) per estrarre opinioni e polarità. | *NLTK/Spacy, Scikit-learn, Pandas* | 🚧 *In lavorazione* |
| **05** | **Prompt Engineering** | Esplorazione delle tecniche di prompting per Large Language Models (LLM) e generazione di output strutturati. | *LLM APIs, Python* | 🚧 *In lavorazione* |
---

## Setup e Installazione

Per configurare l'ambiente di sviluppo e contribuire al progetto, segui questi passaggi.

1.  **Clonare il repository:**
    ```bash
    git clone https://github.com/grannico/Progetti_DataScience_25-26
    cd Progetti_DataScience_25-26
    ```

2.  **Creare un ambiente virtuale:**
    È fondamentale creare un ambiente virtuale (es. `env`) per isolare le dipendenze.
    ```bash
    python3 -m venv env
    ```

3.  **Attivare l'ambiente virtuale:**
    * Su macOS/Linux:
        ```bash
        source env/bin/activate
        ```
    * Su Windows:
        ```bash
        .\env\Scripts\activate
        ```

4.  **Installare le dipendenze:**
    Le dipendenze comuni a tutti i progetti sono elencate nel file `requirements.txt` nella cartella principale.
    ```bash
    pip install -r requirements.txt
    ```
    *(Se un progetto futuro dovesse richiedere dipendenze molto specifiche, valuteremo se creare un `requirements.txt` separato per quella cartella).*

5.  **Avviare Jupyter:**
    Una volta attivate le dipendenze, puoi avviare Jupyter:
    ```bash
    jupyter notebook
    ```
