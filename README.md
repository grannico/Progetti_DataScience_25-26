# Progetti Data Science 2025/2026

Questo repository raccoglierà i progetti e le analisi sviluppate per il corso di Data Science durante l'anno accademico 2025/2026.

## Struttura del Repository

Il repository è organizzato in cartelle, dove ogni cartella rappresenta un singolo progetto o modulo di analisi:

* **`Progetto 1`**: Contiene notebook, script e dati relativi al primo progetto.
* **`[Nome_Progetto_2]/`**: Contiene notebook, script e dati relativi al secondo progetto.
* *(...e così via per i progetti futuri)*

## Stack Tecnologico

Lo stack tecnologico comune utilizzato per i progetti in questo repository include:

* **Python 3.x**
* **Pandas:** Per la manipolazione e l'analisi dei dati.
* **Matplotlib:** Per la creazione di visualizzazioni statiche.
* **Seaborn:** Per la visualizzazione statistica avanzata.
* **Scikit-learn:** Per l'implementazione di modelli di machine learning.
* **Jupyter:** Per l'analisi esplorativa, la prototipazione e la documentazione.

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
