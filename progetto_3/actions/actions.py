from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAZIONE ---
FILENAME = "dataset_ricette_giallozafferano.csv"
CURRENT_DIR = Path(__file__).parent
DATA_FILE = None

possibili_percorsi = [
    CURRENT_DIR / FILENAME,
    CURRENT_DIR.parent / "resources" / FILENAME,
    CURRENT_DIR.parent / FILENAME
]
for p in possibili_percorsi:
    if p.exists():
        DATA_FILE = p
        break

# --- STOPWORDS ---
STOPWORDS = {
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", 
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "o",
    "d", "l", "c", "n", "del", "dello", "della", "dei", "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle",
    "g", "gr", "kg", "ml", "cl", "l", "litro", "cucchiaio", "cucchiaino", "pizzico", "qb", "q.b.", "spicchio", "fetta", "fette", "tazza", "bicchiere", "mazzetto", "rametto", "ciuffo", "foglia", "foglie",
    "fresco", "fresca", "secco", "secca", "vergine", "extra", "extravergine", 
    "macinato", "grattugiato", "spremuto", "tagliato", "tritato", 
    "cubetti", "fettine", "pezzi", "biologico", "interi", "intero", "scaglie", "fiocchi",
    "liquida", "montare", "vegetale", 
    "baccello", "bacca", "stecca", "estratto", "aroma", "fialetta", 
    "polvere", "bustina", "bustine",
    "dop", "igp", "doc", "fino", "grosso", 
    "nero", "bianco", "rosso", "verde", "amaro", "fondente",
    "velo", "vanigliato",
    "maldon", "marino", "iodato",
    "oliva", "semi", "mais", "arachidi",
    "00", "0", "1", "2", "manitoba", "integrale",
    "chimico", "madre", "istantaneo", "birra",
    "romano", "parmigiano", "reggiano", "grana", "padano", "malto"
}

# --- SINONIMI ---
SINONIMI = {
    "uova": ["uova", "uovo", "tuorli", "tuorlo", "albumi", "albume"],
    "pasta": ["pasta", "spaghetti", "rigatoni", "penne", "fusilli", "farfalle", "linguine", "paccheri"],
    "olio": ["olio", "evo", "extravergine"],
    "lievito": ["lievito", "birra"], 
    "carne": ["carne", "manzo", "vitello", "pollo", "tacchino", "salsiccia", "macinato", "maiale", "agnello", "bistecca", "petto", "arista", "lonza"],
    "pesce": ["pesce", "tonno", "salmone", "merluzzo", "orata", "gamberi", "spada", "branzino", "sarde", "alici", "polpo", "calamari"],
    "verdure": ["verdure", "zucchine", "melanzane", "peperoni", "carote", "spinaci", "pomodori", "zucca", "bieta", "insalata"],
    "formaggio": ["formaggio", "parmigiano", "pecorino", "grana", "cacio", "gorgonzola", "provola", "mozzarella", "ricotta"]
}

MAPPA_SINONIMI = {val: key for key, vals in SINONIMI.items() for val in vals}
KEYWORDS_PRIMI = ["pasta", "spaghetti", "riso", "risotto", "gnocchi", "zuppa", "brodo", "vellutata", "minestra", "lasagne", "tortellini", "ravioli", "penne", "fusilli"]
KEYWORDS_DOLCI = ["torta", "biscotti", "crema", "mousse", "tiramisù", "ciambellone", "crostata", "muffin", "gelato", "dolce", "pudding", "plumcake", "bignè"]
DISPENSA_BASE = {"sale", "pepe", "zucchero", "olio", "acqua", "aceto"}

class RecipeEngine:
    def __init__(self, csv_path):
        print("⚙️ Inizializzazione Engine Ricerca...")
        try:
            self.df = pd.read_csv(csv_path)
            # Salvo l'indice per le chiamate ID sicure su Telegram
            self.df['original_id'] = self.df.index 
            self.df['Ingredienti_Clean'] = self.df['Ingredienti'].apply(self._normalize_text)
            self.df['Ingredienti_Set'] = self.df['Ingredienti_Clean'].apply(
                lambda x: set([w for w in x.split() if w not in STOPWORDS])
            )
            self.vectorizer = TfidfVectorizer(min_df=1)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Ingredienti_Clean'])
            print("✅ Engine caricato.")
        except Exception as e:
            print(f"⚠️ Errore critico dataset: {e}")
            self.df = pd.DataFrame()

    def _normalize_text(self, text):
        if pd.isna(text): return ""
        text = text.lower()
        text = re.sub(r'[^a-zàèéìòù ]', ' ', text)
        tokens = text.split()
        tokens = [MAPPA_SINONIMI.get(t, t) for t in tokens]
        return " ".join(tokens)

    def search(self, user_query, tipologia_filter="", top_n=3):
        if self.df.empty: return []

        query_clean = self._normalize_text(user_query)
        user_tokens_real = set([w for w in query_clean.split() if w not in STOPWORDS])
        user_tokens_smart = user_tokens_real.copy()
        user_tokens_smart.update(DISPENSA_BASE)

        def calculate_score(recipe_set):
            match_real = recipe_set & user_tokens_real
            score_real = len(match_real) * 10
            match_dispensa = (recipe_set & DISPENSA_BASE) - user_tokens_real
            score_dispensa = len(match_dispensa) * 1
            missing = recipe_set - user_tokens_smart
            penalty = len(missing) * 2
            total_score = score_real + score_dispensa - penalty
            return total_score, len(match_real), list(missing)

        analysis = self.df['Ingredienti_Set'].apply(calculate_score)
        self.df['Ranking_Score'] = analysis.apply(lambda x: x[0])
        self.df['Matched_Real'] = analysis.apply(lambda x: x[1])
        self.df['Missing_List'] = analysis.apply(lambda x: x[2])
        self.df['Missing_Count'] = self.df['Missing_List'].apply(len)

        mask = pd.Series([True] * len(self.df))
        if tipologia_filter:
            tip = tipologia_filter.lower()
            if "second" in tip: 
                mask = ~self.df['Titolo'].str.lower().str.contains('|'.join(KEYWORDS_PRIMI + KEYWORDS_DOLCI))
            elif "prim" in tip: 
                mask = self.df['Titolo'].str.lower().str.contains('|'.join(KEYWORDS_PRIMI))
            elif "dolc" in tip or "dessert" in tip:
                mask = self.df['Titolo'].str.lower().str.contains('|'.join(KEYWORDS_DOLCI))

        results = self.df[mask].copy()
        results = results.sort_values(by=['Ranking_Score', 'Missing_Count'], ascending=[False, True])
        results = results[results['Matched_Real'] > 0]
        return results.head(top_n)

engine = RecipeEngine(DATA_FILE)

class ActionCercaRicette(Action):
    def name(self) -> Text:
        return "action_cerca_ricette"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1. Recupero Slot
        ingrediente = tracker.get_slot("ingrediente")
        tipologia = tracker.get_slot("tipologia")
        
        # 2. LOGICA "SMART RESET"
        latest_entities = [e['entity'] for e in tracker.latest_message.get('entities', [])]
        
        reset_needed = False
        if "tipologia" in latest_entities and "ingrediente" not in latest_entities:
            print("DEBUG: Cambio contesto rilevato -> Resetto ingredienti vecchi.")
            ingrediente = "" 
            reset_needed = True
        
        # Pulizia
        if isinstance(ingrediente, list): ingrediente = " ".join(ingrediente)
        if isinstance(tipologia, list): tipologia = " ".join(tipologia)
        if not ingrediente: ingrediente = ""
        if not tipologia: tipologia = ""

        query = str(ingrediente)

        # --- GESTIONE MANCANZA INGREDIENTI CON ESEMPIO DINAMICO ---
        if not query:
            if tipologia:
                # Logica per scegliere un esempio sensato
                tipo_low = tipologia.lower()
                ex_ing = "pomodoro" # Fallback generico

                if "prim" in tipo_low or "pasta" in tipo_low or "ris" in tipo_low:
                    ex_ing = "zucchine" # o "pancetta"
                elif "second" in tipo_low or "carn" in tipo_low or "pesc" in tipo_low:
                    ex_ing = "pollo"
                elif "dolc" in tipo_low or "torta" in tipo_low or "dessert" in tipo_low:
                    ex_ing = "cioccolato"
                elif "contorn" in tipo_low or "insalat" in tipo_low:
                    ex_ing = "patate"
                elif "antipast" in tipo_low:
                    ex_ing = "prosciutto"

                dispatcher.utter_message(text=f"Ok, cerchiamo un **{tipologia}**. Ma con quali ingredienti? (es: 'con {ex_ing}')")
            else:
                 dispatcher.utter_message(text="Non ho capito cosa cerchi. Dimmi degli ingredienti! (es. 'ho uova e farina')")
            
            # Applico il reset nello slot di memoria di Rasa
            if reset_needed:
                return [SlotSet("ingrediente", None)]
            return []

        # --- RICERCA REALE ---
        msg = f"👨‍🍳 Cerco ricette"
        if tipologia: msg += f" tipo {tipologia}"
        msg += f" con: {query}..."
        dispatcher.utter_message(text=msg)

        results = engine.search(query, tipologia_filter=tipologia)

        if results.empty:
            dispatcher.utter_message(text="Nessuna ricetta trovata compatibile.")
            return []

        dispatcher.utter_message(text=f"Ecco cosa puoi cucinare:")
        
        for index, row in results.iterrows():
            titolo = row['Titolo']
            ricetta_id = str(row['original_id']) 
            
            tempo = row['Tempo']
            difficolta = row['Difficolta']
            link = row['Link_Originale']
            mancanti = row['Missing_List']
            match_real = row['Matched_Real']
            
            if len(mancanti) == 0:
                msg_extra = "✅ **Hai tutto!**"
            elif len(mancanti) <= 2:
                msg_extra = f"⚠️ Ti manca solo: *{', '.join(mancanti)}*"
            else:
                msg_extra = f"🛒 Ti mancano {len(mancanti)} ingredienti (es. {', '.join(mancanti[:3])}...)"
            
            buttons = [
                {"title": "👨‍🍳 Cucina Ora (Step-by-Step)", "payload": f"/inizia_cucinare{{\"titolo_ricetta\": \"{ricetta_id}\"}}"}
            ]

            messaggio = (
                f"🍽️ **{titolo}** (Usi {match_real} ingredienti tuoi)\n"
                f"⏱️ {tempo} | 📊 {difficolta}\n"
                f"{msg_extra}\n"
                f"🔗 [Leggi Ricetta Originale]({link})"
            )
            dispatcher.utter_message(text=messaggio, buttons=buttons)
        
        if reset_needed:
            return [SlotSet("ingrediente", None)]

        return []

class ActionIniziaCucinare(Action):
    def name(self) -> Text:
        return "action_inizia_cucinare"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        input_value = tracker.get_slot("titolo_ricetta")
        if not input_value:
             input_value = next(tracker.get_latest_entity_values("titolo_ricetta"), None)

        if not input_value:
            dispatcher.utter_message(text="⚠️ Errore tecnico: Non so quale ricetta cucinare.")
            return []
        
        ricetta = pd.DataFrame()
        # Supporto Ibrido: ID Numerico (Bottone) o Titolo (Testo)
        if str(input_value).isdigit():
            rec_id = int(input_value)
            ricetta = engine.df[engine.df['original_id'] == rec_id]
        else:
            ricetta = engine.df[engine.df['Titolo'] == input_value]
        
        if ricetta.empty:
            dispatcher.utter_message(text=f"⚠️ Errore: Non trovo la ricetta richiesta.")
            return []
        
        titolo_reale = ricetta.iloc[0]['Titolo']
        procedimento_raw = ricetta.iloc[0]['Procedimento']
        
        if pd.isna(procedimento_raw):
             dispatcher.utter_message(text="Mi dispiace, non ho il testo del procedimento.")
             return []

        # --- PULIZIA AVANZATA DEI NUMERI (LOOP) ---
        clean_text = " " + " ".join(procedimento_raw.split()) + " "
        unita_protette = r'(?:cm|mm|m|g|gr|kg|ml|cl|l|dl|cucchia|uova|tuorl|album|bicchier|tazz|pizzic|grad|min|ore|°|pezzi|fette|spicchi|persone)'

        # Doppio passaggio per gestire sovrapposizioni (es "1 2 3")
        for _ in range(2):
            clean_text = re.sub(r'\s\d{1,2}\b(?!\s*' + unita_protette + ')', '', clean_text)

        clean_text = " ".join(clean_text.split())

        step_list = re.split(r'\.\s+|\n+', clean_text)
        step_list = [s.strip() for s in step_list if len(s) > 10]
        
        if not step_list:
            dispatcher.utter_message(text="Procedimento troppo breve.")
            return []

        primo_step = step_list[0]
        msg = f"👨‍🍳 **Iniziamo a cucinare: {titolo_reale}**\n\n🔹 **Step 1/{len(step_list)}**\n{primo_step}"
        
        buttons = [{"title": "Fatto! ✅ (Prossimo Step)", "payload": "/step_successivo"}]
        
        dispatcher.utter_message(text=msg, buttons=buttons)

        return [
            SlotSet("lista_step", step_list),
            SlotSet("step_corrente", 0),
            SlotSet("titolo_ricetta_corrente", titolo_reale),
            SlotSet("titolo_ricetta", titolo_reale)
        ]

class ActionStepSuccessivo(Action):
    def name(self) -> Text:
        return "action_step_successivo"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        step_list = tracker.get_slot("lista_step")
        current_index = int(tracker.get_slot("step_corrente") or 0)
        titolo = tracker.get_slot("titolo_ricetta_corrente")
        
        if not step_list:
            dispatcher.utter_message(text="⚠️ Non sto seguendo nessuna ricetta attiva.")
            return []
            
        next_index = current_index + 1
        
        if next_index >= len(step_list):
            # --- MESSAGGIO FINALE + RESET TOTALE ---
            dispatcher.utter_message(text=f"Missione compiuta! 🎉\nLa ricetta **{titolo}** è pronta per essere gustata.\nBuon appetito! 😋")
            
            # Reset totale per permettere una nuova ricerca pulita
            return [
                SlotSet("lista_step", None),
                SlotSet("step_corrente", 0),
                SlotSet("ingrediente", None),         
                SlotSet("tipologia", None),           
                SlotSet("titolo_ricetta", None),      
                SlotSet("titolo_ricetta_corrente", None)
            ]
            
        next_step = step_list[next_index]
        msg = f"🔹 **Step {next_index + 1}/{len(step_list)}**\n{next_step}"
        
        buttons = [{"title": "Fatto! ✅ (Avanti)", "payload": "/step_successivo"}]
        
        dispatcher.utter_message(text=msg, buttons=buttons)
        
        return [SlotSet("step_corrente", next_index)]

class ActionSorprendimi(Action):
    def name(self) -> Text:
        return "action_sorprendimi"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Pesca 1 riga a caso dal DataFrame
        random_row = engine.df.sample(n=1).iloc[0]
        
        titolo = random_row['Titolo']
        ricetta_id = str(random_row['original_id'])
        tempo = random_row['Tempo']
        difficolta = random_row['Difficolta']
        link = random_row['Link_Originale']
        
        # Simuliamo il calcolo ingredienti (ovviamente qui mancheranno tutti, 
        # ma è una proposta random, quindi ci sta!)
        
        buttons = [
            {"title": "👨‍🍳 Cucina Ora (Step-by-Step)", "payload": f"/inizia_cucinare{{\"titolo_ricetta\": \"{ricetta_id}\"}}"}
        ]

        dispatcher.utter_message(text=f"🎲 **Oggi mi sento ispirato! Che ne dici di questa ricetta?**")

        messaggio = (
            f"🍽️ **{titolo}**\n"
            f"⏱️ {tempo} | 📊 {difficolta}\n"
            f"🛒 *Controlla di avere gli ingredienti!*\n"
            f"🔗 [Leggi Ricetta Originale]({link})"
        )
        dispatcher.utter_message(text=messaggio, buttons=buttons)

        return []