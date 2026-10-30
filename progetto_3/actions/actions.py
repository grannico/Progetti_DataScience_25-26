from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import AllSlotsReset, FollowupAction, SlotSet
import pandas as pd
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAZIONE ---
FILENAME = "dataset_ricette_giallozafferano.csv"
CURRENT_DIR = Path(__file__).parent
TYPE_METADATA_FILE = CURRENT_DIR / "tipologie_ricette.json"
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
    "grattugiato", "spremuto", "tagliato", "tritato",
    "cubetti", "fettine", "pezzi", "biologico", "interi", "intero", "scaglie", "fiocchi",
    "liquida", "montare", "vegetale", 
    "baccello", "bacca", "stecca", "estratto", "aroma", "fialetta", 
    "polvere", "bustina", "bustine",
    "dop", "igp", "doc", "fino", "grosso", 
    "nero", "bianco", "rosso", "verde", "amaro", "fondente",
    "velo", "vanigliato",
    "maldon", "marino", "iodato",
    "oliva", "semi",
    "00", "0", "1", "2", "manitoba", "integrale",
    "chimico", "madre", "istantaneo", "birra",
    "romano", "malto"
}

# --- SINONIMI ---
SINONIMI = {
    "uovo": ["uova", "uovo", "tuorli", "tuorlo", "albumi", "albume"],
    "olio": ["olio", "evo", "extravergine"],
    "lievito": ["lievito", "birra"],
    "manzo": ["manzo", "bovino", "bovina"],
    "maiale": ["maiale", "suino", "suina"],
    "gambero": ["gambero", "gamberi"],
    "parmigiano": ["parmigiano", "reggiano", "grana", "padano"],
    "pecorino": ["pecorino", "cacio"],
    "cipolla": ["cipolla", "cipolle", "cipollotto", "cipollotti"],
    "patata": ["patata", "patate"],
    "pomodoro": ["pomodoro", "pomodori"],
    "carota": ["carota", "carote"],
    "melanzana": ["melanzana", "melanzane"],
    "peperone": ["peperone", "peperoni"],
    "fungo": ["fungo", "funghi"],
    "arachide": ["arachide", "arachidi"],
    "zucchina": ["zucchina", "zucchine"],
    "mela": ["mela", "mele"],
    "limone": ["limone", "limoni"],
    "oliva": ["oliva", "olive"],
    "mandorla": ["mandorla", "mandorle"],
    "noce": ["noce", "noci"],
    "fagiolo": ["fagiolo", "fagioli"],
    "cece": ["cece", "ceci"],
}

# Gli ingredienti generici non vengono espansi automaticamente: prima il bot
# chiede quale variante possiede davvero l'utente.
CATEGORIE_GENERICHE = {
    "carne": ["pollo", "manzo", "maiale", "vitello"],
    "pesce": ["salmone", "tonno", "merluzzo", "orata"],
    "verdure": ["zucchine", "melanzane", "peperoni", "spinaci"],
    "formaggio": ["mozzarella", "ricotta", "parmigiano", "gorgonzola"],
    "pasta": ["spaghetti", "penne", "rigatoni", "linguine"],
}

PREFERENZE_ALIMENTARI = {
    "vegetariana": {
        "carne", "pollo", "manzo", "maiale", "vitello", "tacchino",
        "salsiccia", "agnello", "pancetta", "guanciale", "prosciutto",
        "salame", "mortadella", "bresaola", "speck", "tonno", "salmone",
        "merluzzo", "orata", "gambero", "spada", "branzino", "sarde",
        "alici", "polpo", "calamari", "seppia", "seppie", "cozze",
        "vongole", "acciughe",
    },
    "vegana": {
        "carne", "pollo", "manzo", "maiale", "vitello", "tacchino",
        "salsiccia", "agnello", "pancetta", "guanciale", "prosciutto",
        "salame", "mortadella", "bresaola", "speck", "tonno", "salmone",
        "merluzzo", "orata", "gambero", "spada", "branzino", "sarde",
        "alici", "polpo", "calamari", "seppia", "seppie", "cozze",
        "vongole", "acciughe",
        "uovo", "latte", "burro", "panna", "mozzarella", "ricotta",
        "parmigiano", "pecorino", "gorgonzola", "provola", "yogurt",
        "mascarpone", "miele",
    },
    "senza glutine": {
        "farina", "pasta", "spaghetti", "penne", "rigatoni", "linguine",
        "paccheri", "fusilli", "farfalle", "lasagne", "pane",
        "pangrattato", "semola", "couscous", "orzo", "farro", "seitan",
    },
    "senza lattosio": {
        "latte", "burro", "panna", "mozzarella", "ricotta", "parmigiano",
        "pecorino", "gorgonzola", "provola", "yogurt", "mascarpone",
    },
}

GRUPPI_ESCLUSIONE = {
    "carne": {
        "carne", "pollo", "manzo", "maiale", "vitello", "tacchino",
        "salsiccia", "agnello", "pancetta", "guanciale", "prosciutto",
        "salame", "mortadella", "bresaola", "speck",
    },
    "pesce": {
        "tonno", "salmone", "merluzzo", "orata", "gambero", "spada",
        "branzino", "sarde", "alici", "polpo", "calamari", "cozze",
        "vongole", "acciughe",
    },
    "formaggio": {
        "mozzarella", "ricotta", "parmigiano", "pecorino", "gorgonzola",
        "provola", "mascarpone", "fontina", "taleggio", "scamorza",
    },
    "glutine": PREFERENZE_ALIMENTARI["senza glutine"],
    "lattosio": PREFERENZE_ALIMENTARI["senza lattosio"],
}

MAPPA_SINONIMI = {val: key for key, vals in SINONIMI.items() for val in vals}
DISPENSA_BASE = {"sale", "pepe", "zucchero", "olio", "acqua", "aceto"}

# Parole che descrivono forma, taglio, qualità o lavorazione. Vengono tolte
# dalla voce strutturata, ma il testo completo resta disponibile al TF-IDF.
DESCRITTORI_INGREDIENTI = {
    "petto", "petti", "coscia", "cosce", "sovracoscia", "sovracosce",
    "filetto", "filetti", "scorza", "succo", "passata", "polpa",
    "concentrato", "doppio", "pelato", "pelati",
    "dorato", "dorata", "dorati", "dorate",
    "ramato", "ramata", "ramati", "ramate", "ovale", "ovali",
    "rosso", "rossa", "rossi", "rosse",
    "bianco", "bianca", "bianchi", "bianche",
    "nero", "nera", "neri", "nere", "verde", "verdi",
    "giallo", "gialla", "gialli", "gialle", "viola",
    "piccante", "dolce",
    "raffermo", "affumicato", "affumicata", "cotto", "cotta", "crudo",
    "cruda", "macinato", "macinata", "rimacinato", "rimacinata",
    "snocciolato", "snocciolate", "surgelato", "surgelati", "precotto",
    "precotta", "goccia", "gocce", "granella", "tipo",
    "renetta", "renette", "golden", "delica",
}

TIPOLOGIA_ALIASES = {
    "antipasto": "antipasto",
    "antipasti": "antipasto",
    "primo": "primo",
    "primi": "primo",
    "secondo": "secondo",
    "secondi": "secondo",
    "contorno": "contorno",
    "contorni": "contorno",
    "dolce": "dolce",
    "dolci": "dolce",
    "dessert": "dolce",
    "torta": "dolce",
    "torte": "dolce",
    "piatto unico": "piatto unico",
    "lievitato": "lievitato",
    "lievitati": "lievitato",
    "base": "base",
    "bevanda": "bevanda",
    "bevande": "bevanda",
}

PAROLE_INTRODUTTIVE_TITOLO = {
    "ricetta", "ricette", "prepara", "preparare", "preparo", "cucina",
    "cucinare", "voglio", "vorrei", "fammi", "dammi", "darmi",
    "spiegami", "come", "fare", "si", "fa", "sai", "conosci",
    "procedimento", "mostrami", "mostrare", "puoi", "potresti", "mi",
    "serve", "cerco", "cerca", "trova", "trovami",
}


def reset_search_context() -> List[Dict[Text, Any]]:
    """Azzera la conversazione e ripristina i valori iniziali della ricerca."""
    return [
        AllSlotsReset(),
        SlotSet("modalita_ricerca", "guidata"),
        SlotSet("ricerca_in_attesa", False),
        SlotSet("step_corrente", 0),
    ]

class RecipeEngine:
    def __init__(self, csv_path):
        print("Inizializzazione motore di ricerca...")
        try:
            self.df = pd.read_csv(csv_path)
            # Salvo l'indice per le chiamate ID sicure su Telegram
            self.df['original_id'] = self.df.index
            self._load_recipe_types()
            self.df['Ingredienti_Clean'] = self.df['Ingredienti'].apply(self._normalize_text)
            self.df['Ingredienti_Strutturati'] = self.df['Ingredienti'].apply(
                self._structured_ingredients
            )
            # Ingredienti_Set resta il nome usato dal ranking, ma contiene
            # voci canoniche (es. "pollo"), non singole parole scollegate.
            self.df['Ingredienti_Set'] = self.df['Ingredienti_Strutturati'].apply(set)
            self.df['Ingredienti_Tags'] = self.df['Ingredienti_Strutturati'].apply(
                lambda values: {
                    token for value in values for token in value.split()
                }
            )
            self.vectorizer = TfidfVectorizer(min_df=1)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Ingredienti_Clean'])
            print("Motore di ricerca caricato.")
        except Exception as e:
            print(f"Errore durante il caricamento del dataset: {e}")
            self.df = pd.DataFrame()

    def _load_recipe_types(self):
        """Unisce la classificazione manuale per titolo al dataset originale."""
        if "Tipologia" in self.df.columns:
            self.df["Tipologia"] = self.df["Tipologia"].fillna("")
            return

        if TYPE_METADATA_FILE.exists():
            metadata = pd.DataFrame(
                json.loads(TYPE_METADATA_FILE.read_text(encoding="utf-8"))
            )
            self.df = self.df.merge(
                metadata[["Titolo", "Tipologia"]],
                on="Titolo",
                how="left",
                validate="one_to_one",
            )
        else:
            self.df["Tipologia"] = ""

        missing = self.df["Tipologia"].isna() | self.df["Tipologia"].eq("")
        if missing.any():
            missing_titles = self.df.loc[missing, "Titolo"].tolist()
            print(
                "Attenzione: tipologia mancante per "
                f"{len(missing_titles)} ricette."
            )
            self.df.loc[missing, "Tipologia"] = "non classificata"

    def _normalize_text(self, text):
        if pd.isna(text): return ""
        text = text.lower()
        text = re.sub(r'[^a-zàèéìòù ]', ' ', text)
        tokens = text.split()
        tokens = [MAPPA_SINONIMI.get(t, t) for t in tokens]
        return " ".join(tokens)

    def _canonical_ingredient(self, value):
        """Riduce una voce CSV a un ingrediente confrontabile e leggibile."""
        normalized = self._normalize_text(value)
        raw_tokens = normalized.split()
        tokens = []
        for token in raw_tokens:
            # "oliva" è rumore in "olio d'oliva", ma è un ingrediente vero
            # quando compare da solo come oliva/olive.
            if token == "oliva" and "olio" not in raw_tokens:
                tokens.append(token)
                continue
            if token in STOPWORDS or token in DESCRITTORI_INGREDIENTI:
                continue
            if token not in tokens:
                tokens.append(token)
        return " ".join(tokens)

    def _structured_ingredients(self, value):
        """Divide il campo CSV in voci: una voce equivale a un ingrediente."""
        if pd.isna(value):
            return []
        phrases = re.split(r'\s*[,;]\s*', str(value))
        canonical = [self._canonical_ingredient(phrase) for phrase in phrases]
        return list(dict.fromkeys(value for value in canonical if value))

    def _query_ingredients(self, user_query, user_ingredients=None):
        if user_ingredients:
            values = user_ingredients
        else:
            # Le custom action passa sempre la lista di entità. Questo fallback
            # conserva l'uso diretto del motore nei test e negli script.
            values = re.split(r'\s*[,;]\s*|\s+', str(user_query or ""))
        canonical = [self._canonical_ingredient(value) for value in values]
        return {value for value in canonical if value}

    @staticmethod
    def _ingredients_compatible(requested, recipe_ingredient):
        """Confronta due voci senza scomporle nel conteggio degli ingredienti."""
        requested_tokens = set(str(requested).split())
        recipe_tokens = set(str(recipe_ingredient).split())
        if not requested_tokens or not recipe_tokens:
            return False
        return (
            requested_tokens.issubset(recipe_tokens)
            or recipe_tokens.issubset(requested_tokens)
        )

    @classmethod
    def _match_ingredient_sets(cls, available, required):
        """Abbina al massimo una voce disponibile a ogni ingrediente richiesto."""
        remaining = set(required)
        matched_available = set()
        matched_required = set()

        # Prima i valori piu specifici; a parita, prima il match esatto.
        for value in sorted(available, key=lambda item: (-len(item.split()), item)):
            candidates = [
                ingredient for ingredient in remaining
                if cls._ingredients_compatible(value, ingredient)
            ]
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda ingredient: (
                    ingredient != value,
                    abs(len(ingredient.split()) - len(value.split())),
                    ingredient,
                ),
            )
            matched_available.add(value)
            matched_required.add(selected)
            remaining.remove(selected)

        return matched_available, matched_required

    @staticmethod
    def _title_signature(value):
        normalized = str(value or "").casefold()
        normalized = re.sub(r'[^a-zàèéìòù0-9 ]', ' ', normalized)
        tokens = [word for word in normalized.split() if word not in STOPWORDS]
        return " ".join(tokens)

    def resolve_title(self, user_text, allow_fuzzy=True):
        """Riconosce un titolo noto anche quando l'NLU non estrae l'entità."""
        text_signature = self._title_signature(user_text)
        if not text_signature:
            return ""

        exact_matches = []
        for title in self.df['Titolo']:
            signature = self._title_signature(title)
            if signature and f" {signature} " in f" {text_signature} ":
                exact_matches.append((len(signature.split()), title))
        if exact_matches:
            return max(exact_matches, key=lambda value: value[0])[1]

        if not allow_fuzzy:
            return ""

        request_tokens = [
            token for token in text_signature.split()
            if token not in PAROLE_INTRODUTTIVE_TITOLO
        ]
        candidate = " ".join(request_tokens)
        if not candidate:
            return ""

        # Gestisce nomi d'uso comune più brevi del titolo completo, come
        # "carbonara" per "Spaghetti alla Carbonara", soltanto se il match è
        # univoco nel catalogo.
        candidate_set = set(request_tokens)
        partial_matches = [
            title for title in self.df['Titolo']
            if candidate_set.issubset(
                set(self._title_signature(title).split())
            )
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]

        # Il confronto approssimativo avviene sui singoli termini. In questo
        # modo "carbonaraa" può corrispondere a "carbonara", mentre una frase
        # estranea viene scartata appena contiene parole lontane dal titolo.
        scored = []
        for title in self.df['Titolo']:
            title_tokens = self._title_signature(title).split()
            if not title_tokens:
                continue

            token_scores = [
                max(
                    SequenceMatcher(None, token, title_token).ratio()
                    for title_token in title_tokens
                )
                for token in request_tokens
            ]
            minimum_score = min(token_scores)
            average_score = sum(token_scores) / len(token_scores)

            if len(request_tokens) == 1:
                valid = len(request_tokens[0]) >= 5 and minimum_score >= 0.88
            else:
                valid = minimum_score >= 0.80 and average_score >= 0.88
            if not valid:
                continue

            exact_terms = sum(
                1 for token in request_tokens if token in title_tokens
            )
            ranking_score = (
                average_score
                + exact_terms * 0.02
                - max(0, len(title_tokens) - len(request_tokens)) * 0.005
            )
            scored.append((ranking_score, title))

        if not scored:
            return ""

        scored.sort(reverse=True)
        best_score, best_title = scored[0]
        if len(scored) > 1 and best_score - scored[1][0] < 0.03:
            return ""
        return best_title

    @staticmethod
    def _duration_to_minutes(value):
        """Converte valori come '40 min' o '1 h 30 min' in minuti."""
        if pd.isna(value):
            return None

        value = str(value).lower().strip()
        hours_match = re.search(r'(\d+)\s*h', value)
        minutes_match = re.search(r'(\d+)\s*min', value)

        if not hours_match and not minutes_match:
            return None

        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        return hours * 60 + minutes

    @staticmethod
    def _requested_max_minutes(tempo_filter):
        """Interpreta il filtro temporale estratto da Rasa."""
        if not tempo_filter:
            return None

        value = str(tempo_filter).lower()
        hours_match = re.search(r'(\d+)\s*(?:h|ora|ore)', value)
        minutes_match = re.search(r'(\d+)\s*(?:min|minuti|minuto)', value)
        if hours_match or minutes_match:
            hours = int(hours_match.group(1)) if hours_match else 0
            minutes = int(minutes_match.group(1)) if minutes_match else 0
            return hours * 60 + minutes

        number_match = re.search(r'(\d+)', value)
        if number_match:
            return int(number_match.group(1))

        if any(word in value for word in ["veloc", "rapid", "poco tempo", "breve"]):
            return 30

        return None

    @staticmethod
    def _difficulty_mask(series, difficolta_filter):
        """Costruisce il filtro sulle categorie presenti nel dataset."""
        value = str(difficolta_filter or "").lower()
        normalized = series.fillna("").str.lower()

        if "facil" in value:
            return normalized.isin(["facile", "molto facile"])
        if "medi" in value:
            return normalized.eq("media")
        if "diffic" in value or "impegn" in value:
            return normalized.isin(["difficile", "molto difficile"])

        return pd.Series(True, index=series.index)

    def _apply_constraints(self, results, tipologia_filter="",
                           difficolta_filter="", tempo_filter="",
                           excluded_ingredients=None,
                           dietary_preferences=None):
        """Applica a un insieme di ricette tutti i vincoli dell'utente."""
        mask = pd.Series(True, index=results.index)

        if tipologia_filter:
            requested_type = TIPOLOGIA_ALIASES.get(
                str(tipologia_filter).casefold().strip(), ""
            )
            if requested_type:
                mask &= results['Tipologia'].str.casefold().eq(requested_type)

        if difficolta_filter:
            mask &= self._difficulty_mask(
                results['Difficolta'], difficolta_filter
            )

        max_minutes = self._requested_max_minutes(tempo_filter)
        if max_minutes is not None:
            durations = results['Tempo'].apply(self._duration_to_minutes)
            mask &= durations.notna() & durations.le(max_minutes)

        if excluded_ingredients:
            excluded_tokens = set()
            for ingredient in excluded_ingredients:
                normalized = self._normalize_text(ingredient)
                for word in normalized.split():
                    if word not in STOPWORDS:
                        excluded_tokens.update(
                            GRUPPI_ESCLUSIONE.get(word, {word})
                        )
            if excluded_tokens:
                mask &= ~results['Ingredienti_Tags'].apply(
                    lambda recipe_tags: bool(recipe_tags & excluded_tokens)
                )

        for preference in dietary_preferences or []:
            forbidden_tokens = PREFERENZE_ALIMENTARI.get(preference, set())
            if forbidden_tokens:
                mask &= ~results['Ingredienti_Tags'].apply(
                    lambda recipe_tags: bool(recipe_tags & forbidden_tokens)
                )

        return results[mask].copy()

    def compatible_recipes(self, tipologia_filter="", difficolta_filter="",
                           tempo_filter="", excluded_ingredients=None,
                           dietary_preferences=None):
        """Restituisce le ricette compatibili, anche senza una query ingredienti."""
        if self.df.empty:
            return self.df.copy()
        return self._apply_constraints(
            self.df.copy(),
            tipologia_filter=tipologia_filter,
            difficolta_filter=difficolta_filter,
            tempo_filter=tempo_filter,
            excluded_ingredients=excluded_ingredients,
            dietary_preferences=dietary_preferences,
        )

    def constraint_conflicts(self, recipes, tipologia_filter="",
                             difficolta_filter="", tempo_filter="",
                             excluded_ingredients=None,
                             dietary_preferences=None):
        """Indica quali vincoli, presi singolarmente, escludono una ricetta."""
        conflicts = []

        for preference in dietary_preferences or []:
            if self._apply_constraints(
                recipes, dietary_preferences=[preference]
            ).empty:
                conflicts.append(("preferenza", preference))

        for ingredient in excluded_ingredients or []:
            if self._apply_constraints(
                recipes, excluded_ingredients=[ingredient]
            ).empty:
                conflicts.append(("esclusione", ingredient))

        if tipologia_filter and self._apply_constraints(
            recipes, tipologia_filter=tipologia_filter
        ).empty:
            conflicts.append(("tipologia", tipologia_filter))

        if difficolta_filter and self._apply_constraints(
            recipes, difficolta_filter=difficolta_filter
        ).empty:
            conflicts.append(("difficolta", difficolta_filter))

        if tempo_filter and self._apply_constraints(
            recipes, tempo_filter=tempo_filter
        ).empty:
            conflicts.append(("tempo", tempo_filter))

        return conflicts

    def search_by_title(self, title_query, tipologia_filter="",
                        difficolta_filter="", tempo_filter="",
                        excluded_ingredients=None,
                        dietary_preferences=None, top_n=3):
        """Cerca una ricetta per titolo e applica gli stessi vincoli del motore."""
        if self.df.empty:
            return self.df.copy()

        query_clean = self._normalize_text(title_query)
        query_tokens = [
            word for word in query_clean.split() if word not in STOPWORDS
        ]
        if not query_tokens:
            return self.df.iloc[0:0].copy()

        query_signature = " ".join(query_tokens)
        results = self.compatible_recipes(
            tipologia_filter=tipologia_filter,
            difficolta_filter=difficolta_filter,
            tempo_filter=tempo_filter,
            excluded_ingredients=excluded_ingredients,
            dietary_preferences=dietary_preferences,
        )
        if results.empty:
            return results

        def title_score(title):
            normalized = self._normalize_text(title)
            title_tokens = [
                word for word in normalized.split() if word not in STOPWORDS
            ]
            title_set = set(title_tokens)
            matched = sum(1 for word in query_tokens if word in title_set)
            coverage = matched / len(query_tokens)
            title_signature = " ".join(title_tokens)
            exact = title_signature == query_signature
            contains = query_signature in title_signature
            extra_words = max(0, len(title_tokens) - len(query_tokens))
            score = coverage * 50 - extra_words
            if contains:
                score += 30
            if exact:
                score += 50
            return score, coverage

        title_analysis = results['Titolo'].apply(title_score)
        results['Title_Score'] = title_analysis.apply(lambda value: value[0])
        results['Title_Coverage'] = title_analysis.apply(lambda value: value[1])
        results = results[results['Title_Coverage'].ge(1.0)]
        return results.sort_values(
            by=['Title_Score'], ascending=False
        ).head(top_n)

    def search(self, user_query, tipologia_filter="", difficolta_filter="",
               tempo_filter="", excluded_ingredients=None,
               dietary_preferences=None, top_n=3, user_ingredients=None):
        if self.df.empty:
            return self.df.copy()

        query_clean = self._normalize_text(user_query)
        user_values_real = self._query_ingredients(
            user_query, user_ingredients=user_ingredients
        )

        def calculate_score(recipe_set):
            match_real, satisfied_by_user = self._match_ingredient_sets(
                user_values_real, recipe_set
            )
            remaining_recipe = recipe_set - satisfied_by_user
            _, satisfied_by_pantry = self._match_ingredient_sets(
                DISPENSA_BASE, remaining_recipe
            )
            score_real = len(match_real) * 10
            score_dispensa = len(satisfied_by_pantry)
            missing = remaining_recipe - satisfied_by_pantry
            penalty = len(missing) * 2
            total_score = score_real + score_dispensa - penalty
            unmatched_user = user_values_real - match_real
            coverage = (
                len(match_real) / len(user_values_real)
                if user_values_real else 0
            )
            return (
                total_score,
                len(match_real),
                sorted(match_real),
                sorted(unmatched_user),
                coverage,
                sorted(missing),
            )

        analysis = self.df['Ingredienti_Set'].apply(calculate_score)
        results = self.df.copy()
        results['Heuristic_Score'] = analysis.apply(lambda x: x[0])
        results['Matched_Real'] = analysis.apply(lambda x: x[1])
        results['Matched_Ingredients'] = analysis.apply(lambda x: x[2])
        results['Unmatched_User_Ingredients'] = analysis.apply(lambda x: x[3])
        results['Ingredient_Coverage'] = analysis.apply(lambda x: x[4])
        results['Missing_List'] = analysis.apply(lambda x: x[5])
        results['Missing_Count'] = results['Missing_List'].apply(len)

        # La similarità TF-IDF affianca il punteggio basato sulle corrispondenze.
        # Il fattore 10 porta il valore [0, 1] sulla stessa scala di un match reale.
        query_vector = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        results['TFIDF_Similarity'] = similarities
        results['Ranking_Score'] = (
            results['Heuristic_Score'] + results['TFIDF_Similarity'] * 10
        )

        results = self._apply_constraints(
            results,
            tipologia_filter=tipologia_filter,
            difficolta_filter=difficolta_filter,
            tempo_filter=tempo_filter,
            excluded_ingredients=excluded_ingredients,
            dietary_preferences=dietary_preferences,
        )
        # Una ricetta che usa tutti gli ingredienti richiesti deve precedere
        # una ricetta che ne usa soltanto una parte.
        results = results.sort_values(
            by=['Ingredient_Coverage', 'Ranking_Score', 'Missing_Count'],
            ascending=[False, False, True]
        )
        results = results[results['Matched_Real'] > 0]
        return results.head(top_n)

engine = RecipeEngine(DATA_FILE)

class ActionCercaRicette(Action):
    def name(self) -> Text:
        return "action_cerca_ricette"

    @staticmethod
    def _latest_entity_values(tracker: Tracker, entity_name: Text) -> List[Text]:
        """Restituisce solo le entita presenti nell'ultimo messaggio."""
        values = []
        for entity in tracker.latest_message.get("entities", []):
            if entity.get("entity") == entity_name and entity.get("value"):
                values.append(str(entity["value"]).strip())
        return values

    @staticmethod
    def _unique_values(values: List[Text]) -> List[Text]:
        """Rimuove i duplicati conservando l'ordine di inserimento."""
        unique = []
        seen = set()
        for value in values:
            normalized = value.casefold()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(value)
        return unique

    @staticmethod
    def _natural_list(values: List[Text]) -> Text:
        """Formatta una breve lista in modo naturale per la risposta."""
        values = [str(value) for value in values if value]
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        return f"{', '.join(values[:-1])} e {values[-1]}"

    @staticmethod
    def _single_ingredient_buttons(
        ingredients: List[Text]
    ) -> List[Dict[Text, Text]]:
        """Crea alternative esplicite per una ricerca risultata parziale."""
        buttons = []
        for ingredient in ingredients[:2]:
            payload = "/cerca_ricetta" + json.dumps(
                {"ingrediente": ingredient}, ensure_ascii=False
            )
            buttons.append({
                "title": f"Cerca solo con {ingredient}",
                "payload": payload,
            })
        return buttons

    @staticmethod
    def _clarification_buttons(generic_ingredient: Text) -> List[Dict[Text, Text]]:
        """Propone varianti concrete per un ingrediente troppo generico."""
        buttons = []
        for option in CATEGORIE_GENERICHE.get(generic_ingredient, []):
            payload = "/fornisce_ingredienti" + json.dumps(
                {"ingrediente": option}, ensure_ascii=False
            )
            buttons.append({"title": option.capitalize(), "payload": payload})
        return buttons

    @staticmethod
    def _suggest_missing_ingredient(results: pd.DataFrame) -> Text:
        """Trova l'ingrediente aggiuntivo comune a più risultati mostrati."""
        frequencies = Counter()
        for missing_values in results['Missing_List']:
            useful_values = {
                value for value in missing_values
                if len(value) > 2 and value not in CATEGORIE_GENERICHE
            }
            frequencies.update(useful_values)

        if not frequencies:
            return ""

        ingredient, frequency = frequencies.most_common(1)[0]
        return ingredient if frequency >= 2 else ""

    @staticmethod
    def _normalize_search_mode(value: Text) -> Text:
        value = str(value or "").casefold()
        if any(term in value for term in [
            "rigoros", "solo ciò", "solo cio", "solo quello",
            "non voglio comprare", "non compro",
        ]):
            return "rigorosa"
        if any(term in value for term in [
            "flessibil", "uno o due", "1 o 2", "comprare", "paio",
            "qualche ingrediente",
        ]):
            return "flessibile"
        if any(term in value for term in [
            "liber", "senza limiti", "idea", "approssimativ",
        ]):
            return "libera"
        return ""

    @staticmethod
    def _normalize_preferences(values: List[Text]) -> List[Text]:
        normalized = []
        for value in values:
            lowered = str(value).casefold()
            if "nessun" in lowered or "rimuovi" in lowered:
                return []
            if "vegan" in lowered:
                normalized.append("vegana")
            elif "vegetarian" in lowered:
                normalized.append("vegetariana")
            elif "glutin" in lowered or "celiac" in lowered:
                normalized.append("senza glutine")
            elif "lattos" in lowered:
                normalized.append("senza lattosio")
        return ActionCercaRicette._unique_values(normalized)

    @staticmethod
    def _preference_ingredient_conflicts(
        preferences: List[Text], ingredients: List[Text]
    ) -> Dict[Text, List[Text]]:
        """Individua ingredienti incompatibili con i vincoli alimentari."""
        conflicts = {}
        for preference in preferences:
            forbidden = PREFERENZE_ALIMENTARI.get(preference, set())
            if not forbidden:
                continue

            # Una categoria generica e incompatibile quando tutte le varianti
            # proposte dal bot violano il vincolo. Per esempio, "carne" e
            # "pesce" sono incompatibili con una preferenza vegetariana.
            forbidden_categories = {
                category
                for category, options in CATEGORIE_GENERICHE.items()
                if options and all(
                    bool(
                        set(engine._canonical_ingredient(option).split())
                        & forbidden
                    )
                    for option in options
                )
            }

            incompatible = []
            for ingredient in ingredients:
                canonical = engine._canonical_ingredient(ingredient)
                tokens = set(canonical.split())
                if tokens & forbidden or canonical in forbidden_categories:
                    incompatible.append(ingredient)

            if incompatible:
                conflicts[preference] = ActionCercaRicette._unique_values(
                    incompatible
                )
        return conflicts

    @staticmethod
    def _preference_conflict_buttons(
        multiple_preferences: bool = False,
    ) -> List[Dict[Text, Text]]:
        remove_label = (
            "Rimuovi le preferenze"
            if multiple_preferences
            else "Rimuovi la preferenza"
        )
        return [
            {
                "title": remove_label,
                # Telegram limita callback_data a 64 byte. L'intento senza
                # entita viene interpretato come conferma della rimozione.
                "payload": "/imposta_preferenza",
            },
            {
                "title": "🔄 Nuova ricerca",
                "payload": "/nuova_ricerca",
            },
        ]

    def _preference_conflict_message(
        self, conflicts: Dict[Text, List[Text]]
    ) -> Text:
        criterion_labels = {
            "vegetariana": "la preferenza vegetariana",
            "vegana": "la preferenza vegana",
            "senza glutine": "il vincolo senza glutine",
            "senza lattosio": "il vincolo senza lattosio",
        }
        descriptions = []
        for preference, ingredients in conflicts.items():
            ingredient_label = self._natural_list(ingredients)
            verb = (
                "non è compatibile"
                if len(ingredients) == 1
                else "non sono compatibili"
            )
            criterion = criterion_labels.get(
                preference, f"il criterio {preference}"
            )
            descriptions.append(
                f"{ingredient_label} {verb} con {criterion}"
            )

        return (
            f"Ho rilevato un conflitto: {self._natural_list(descriptions)}. "
            "Puoi rimuovere la preferenza oppure iniziare una nuova ricerca."
        )

    @staticmethod
    def _mode_buttons() -> List[Dict[Text, Text]]:
        return [
            {
                "title": "Solo ciò che ho",
                "payload": '/imposta_modalita{"modalita_ricerca":"rigorosa"}',
            },
            {
                "title": "Compro 1-2 ingredienti",
                "payload": '/imposta_modalita{"modalita_ricerca":"flessibile"}',
            },
            {
                "title": "Ricerca libera",
                "payload": '/imposta_modalita{"modalita_ricerca":"libera"}',
            },
        ]

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Rasa sostituisce lo slot list con le entita dell'ultimo messaggio.
        # Manteniamo quindi una memoria separata per poter gestire frasi come
        # "ho anche il formaggio" senza perdere gli ingredienti precedenti.
        latest_intent = tracker.latest_message.get("intent", {}).get("name", "")
        latest_text = tracker.latest_message.get("text", "")
        latest_ingredients = self._latest_entity_values(tracker, "ingrediente")
        latest_tipologia = self._latest_entity_values(tracker, "tipologia")
        latest_titles = self._latest_entity_values(tracker, "titolo_ricetta")
        latest_difficolta = self._latest_entity_values(tracker, "difficolta")
        latest_tempo = self._latest_entity_values(tracker, "tempo_req")
        latest_exclusions = self._latest_entity_values(
            tracker, "ingrediente_escluso"
        )
        latest_modes = self._latest_entity_values(tracker, "modalita_ricerca")
        latest_preferences = self._latest_entity_values(
            tracker, "preferenza_alimentare"
        )

        stored_ingredients = tracker.get_slot("ingredienti_memoria") or []
        if isinstance(stored_ingredients, str):
            stored_ingredients = [stored_ingredients]

        waiting_for_ingredients = bool(tracker.get_slot("ricerca_in_attesa"))
        pending_generic = tracker.get_slot("ingrediente_da_chiarire") or ""
        suggested_ingredient = tracker.get_slot("ingrediente_suggerito") or ""
        excluded_ingredients = tracker.get_slot("ingredienti_esclusi") or []
        if isinstance(excluded_ingredients, str):
            excluded_ingredients = [excluded_ingredients]
        search_mode = tracker.get_slot("modalita_ricerca") or "guidata"
        dietary_preferences = tracker.get_slot("preferenze_alimentari") or []
        if isinstance(dietary_preferences, str):
            dietary_preferences = [dietary_preferences]

        requested_mode = self._normalize_search_mode(
            latest_modes[-1] if latest_modes
            else tracker.latest_message.get("text", "")
        )
        requested_preferences = self._normalize_preferences(
            latest_preferences
        )
        remove_preferences_button = (
            latest_intent == "imposta_preferenza"
            and latest_text.strip() == "/imposta_preferenza"
        )
        remove_preferences_requested = (
            remove_preferences_button
            or bool(latest_preferences and not requested_preferences)
        )
        resolved_entity_title = (
            engine.resolve_title(latest_titles[-1], allow_fuzzy=True)
            if latest_titles else ""
        )
        resolved_text_title = engine.resolve_title(
            latest_text,
            allow_fuzzy=latest_intent in {
                "cerca_ricetta",
                "cerca_ricetta_titolo",
                "nlu_fallback",
            },
        )
        # Il confronto col catalogo ha priorità: evita che un'entità parziale
        # come "alla" sostituisca un titolo completo riconosciuto nel testo.
        requested_title = (
            resolved_entity_title
            or resolved_text_title
            or (latest_titles[-1] if latest_titles else "")
        )

        if latest_intent == "nega_ingrediente":
            denied_ingredients = self._unique_values(
                latest_exclusions
                + ([suggested_ingredient] if suggested_ingredient else [])
            )
            denied_normalized = {
                value.casefold() for value in denied_ingredients
            }
            added_ingredients = [
                value for value in latest_ingredients
                if value.casefold() not in denied_normalized
            ]
            ingredienti = self._unique_values(
                [
                    value for value in stored_ingredients
                    if value.casefold() not in denied_normalized
                ]
                + added_ingredients
            )

            added_normalized = {
                value.casefold() for value in added_ingredients
            }
            excluded_ingredients = self._unique_values(
                [
                    value for value in excluded_ingredients
                    if value.casefold() not in added_normalized
                ]
                + denied_ingredients
            )

            if denied_ingredients and added_ingredients:
                dispatcher.utter_message(
                    text=(
                        f"Va bene, escludo {self._natural_list(denied_ingredients)} "
                        f"e aggiungo {self._natural_list(added_ingredients)} "
                        "agli ingredienti disponibili."
                    )
                )
            elif denied_ingredients:
                dispatcher.utter_message(
                    text=(
                        f"Va bene, escludo "
                        f"{self._natural_list(denied_ingredients)} e cerco "
                        "un'alternativa con quello che hai."
                    )
                )
            elif added_ingredients:
                dispatcher.utter_message(
                    text=(
                        f"Va bene, aggiungo "
                        f"{self._natural_list(added_ingredients)} agli "
                        "ingredienti disponibili."
                    )
                )
            else:
                dispatcher.utter_message(
                    text="Va bene. Continuiamo con gli ingredienti già indicati."
                )
            tipologia = tracker.get_slot("tipologia") or ""
            difficolta = tracker.get_slot("difficolta") or ""
            tempo_req = tracker.get_slot("tempo_req") or ""
        elif latest_intent == "escludi_ingrediente":
            ingredienti = self._unique_values(stored_ingredients)
            excluded_ingredients = self._unique_values(
                excluded_ingredients + latest_exclusions
            )
            tipologia = tracker.get_slot("tipologia") or ""
            difficolta = tracker.get_slot("difficolta") or ""
            tempo_req = tracker.get_slot("tempo_req") or ""
            if latest_exclusions:
                dispatcher.utter_message(
                    text=(
                        "Va bene, escludo "
                        f"{self._natural_list(latest_exclusions)} dalla ricerca."
                    )
                )
        elif latest_intent == "imposta_modalita":
            ingredienti = self._unique_values(stored_ingredients)
            search_mode = requested_mode or search_mode
            tipologia = tracker.get_slot("tipologia") or ""
            difficolta = tracker.get_slot("difficolta") or ""
            tempo_req = tracker.get_slot("tempo_req") or ""
            dispatcher.utter_message(
                text=f"Modalità di ricerca impostata su {search_mode}."
            )
        elif latest_intent == "imposta_preferenza":
            # Una preferenza e gli ingredienti possono essere comunicati nello
            # stesso messaggio, ad esempio: "sono vegetariano, ho patate".
            ingredienti = self._unique_values(
                stored_ingredients + latest_ingredients
            )
            latest_normalized = {
                ingredient.casefold() for ingredient in latest_ingredients
            }
            excluded_ingredients = [
                value for value in excluded_ingredients
                if value.casefold() not in latest_normalized
            ]
            if remove_preferences_requested:
                dietary_preferences = []
                # Nel conflitto relativo a un titolo, il pulsante deve
                # ripetere la stessa ricerca dopo aver rimosso il vincolo.
                if remove_preferences_button:
                    requested_title = tracker.get_slot("titolo_ricetta") or ""
            elif latest_preferences:
                dietary_preferences = self._unique_values(
                    dietary_preferences + requested_preferences
                )
            tipologia = tracker.get_slot("tipologia") or ""
            difficolta = tracker.get_slot("difficolta") or ""
            tempo_req = tracker.get_slot("tempo_req") or ""
            if dietary_preferences:
                preference_label = (
                    "della preferenza"
                    if len(dietary_preferences) == 1
                    else "delle preferenze"
                )
                dispatcher.utter_message(
                    text=(
                        f"Terrò conto {preference_label} "
                        f"{self._natural_list(dietary_preferences)}."
                    )
                )
            else:
                dispatcher.utter_message(
                    text="Va bene, ho rimosso le preferenze alimentari."
                )
        elif latest_intent == "fornisce_ingredienti":
            # Una risposta affermativa può riferirsi all'ingrediente appena
            # proposto senza ripeterne il nome, ad esempio "ho anche
            # quell'ingrediente".
            if not latest_ingredients and suggested_ingredient:
                latest_ingredients = [suggested_ingredient]

            # Se il bot aveva chiesto di specificare una categoria generica,
            # una risposta valida sostituisce la categoria invece di aggiungersi.
            valid_options = CATEGORIE_GENERICHE.get(pending_generic, [])
            selected_specific = next(
                (
                    value for value in latest_ingredients
                    if value.casefold() in valid_options
                ),
                None,
            )
            if selected_specific:
                stored_ingredients = [
                    value for value in stored_ingredients
                    if value.casefold() != pending_generic.casefold()
                ]
                pending_generic = ""

            ingredienti = self._unique_values(
                stored_ingredients + latest_ingredients
            )
            latest_normalized = {
                ingredient.casefold() for ingredient in latest_ingredients
            }
            excluded_ingredients = [
                value for value in excluded_ingredients
                if value.casefold() not in latest_normalized
            ]
            excluded_ingredients = self._unique_values(
                excluded_ingredients + latest_exclusions
            )
            if requested_mode:
                search_mode = requested_mode
            if latest_preferences:
                dietary_preferences = self._unique_values(
                    dietary_preferences + requested_preferences
                )
            tipologia = latest_tipologia[-1] if latest_tipologia else (tracker.get_slot("tipologia") or "")
            difficolta = latest_difficolta[-1] if latest_difficolta else (tracker.get_slot("difficolta") or "")
            tempo_req = latest_tempo[-1] if latest_tempo else (tracker.get_slot("tempo_req") or "")
        else:
            # Una normale richiesta avvia una nuova ricerca. I vecchi filtri
            # vengono conservati soltanto se il bot stava aspettando gli
            # ingredienti richiesti nel turno precedente.
            ingredienti = self._unique_values(latest_ingredients)
            pending_generic = ""
            excluded_ingredients = self._unique_values(latest_exclusions)
            if requested_mode:
                search_mode = requested_mode
            if latest_preferences:
                dietary_preferences = self._unique_values(
                    dietary_preferences + requested_preferences
                )
            if waiting_for_ingredients:
                tipologia = latest_tipologia[-1] if latest_tipologia else (tracker.get_slot("tipologia") or "")
                difficolta = latest_difficolta[-1] if latest_difficolta else (tracker.get_slot("difficolta") or "")
                tempo_req = latest_tempo[-1] if latest_tempo else (tracker.get_slot("tempo_req") or "")
            else:
                tipologia = latest_tipologia[-1] if latest_tipologia else ""
                difficolta = latest_difficolta[-1] if latest_difficolta else ""
                tempo_req = latest_tempo[-1] if latest_tempo else ""

        generic_to_clarify = next(
            (
                value.casefold() for value in ingredienti
                if value.casefold() in CATEGORIE_GENERICHE
            ),
            "",
        )

        query = " ".join(ingredienti)
        state_events = [
            SlotSet("ingrediente", ingredienti or None),
            SlotSet("ingredienti_memoria", ingredienti or None),
            SlotSet(
                "ingrediente_da_chiarire",
                generic_to_clarify or None,
            ),
            SlotSet("ingrediente_suggerito", None),
            SlotSet("ingredienti_esclusi", excluded_ingredients or None),
            SlotSet("modalita_ricerca", search_mode),
            SlotSet(
                "preferenze_alimentari",
                dietary_preferences or None,
            ),
            SlotSet("tipologia", tipologia or None),
            SlotSet("difficolta", difficolta or None),
            SlotSet("tempo_req", tempo_req or None),
            SlotSet("titolo_ricetta", requested_title or None),
            SlotSet("ultime_ricette_proposte", None),
        ]

        if requested_title:
            dispatcher.utter_message(
                text=f"Cerco nel ricettario: {requested_title}..."
            )
            catalogue_results = engine.search_by_title(
                requested_title, top_n=3
            )

            if catalogue_results.empty:
                dispatcher.utter_message(
                    text=(
                        f"Non ho trovato una ricetta intitolata “{requested_title}” "
                        "nel catalogo. Prova a scrivere un titolo più breve."
                    ),
                    buttons=[{
                        "title": "🔄 Nuova ricerca",
                        "payload": "/nuova_ricerca",
                    }],
                )
                return state_events + [SlotSet("ricerca_in_attesa", False)]

            title_results = engine._apply_constraints(
                catalogue_results,
                tipologia_filter=tipologia,
                difficolta_filter=difficolta,
                tempo_filter=tempo_req,
                excluded_ingredients=excluded_ingredients,
                dietary_preferences=dietary_preferences,
            )

            if title_results.empty:
                conflicts = engine.constraint_conflicts(
                    catalogue_results,
                    tipologia_filter=tipologia,
                    difficolta_filter=difficolta,
                    tempo_filter=tempo_req,
                    excluded_ingredients=excluded_ingredients,
                    dietary_preferences=dietary_preferences,
                )
                conflict_descriptions = []
                for conflict_type, value in conflicts:
                    if conflict_type == "preferenza":
                        conflict_descriptions.append(f"la preferenza {value}")
                    elif conflict_type == "esclusione":
                        conflict_descriptions.append(f"l'esclusione di {value}")
                    elif conflict_type == "tipologia":
                        conflict_descriptions.append(f"la tipologia {value}")
                    elif conflict_type == "difficolta":
                        conflict_descriptions.append(f"la difficoltà {value}")
                    else:
                        conflict_descriptions.append(
                            f"il limite di preparazione {value}"
                        )

                title_found = str(catalogue_results.iloc[0]["Titolo"])
                active_conflicts = self._natural_list(conflict_descriptions)
                if active_conflicts:
                    adjective = (
                        "attivo"
                        if len(conflict_descriptions) == 1
                        and conflicts[0][0] == "tempo"
                        else (
                            "attiva" if len(conflict_descriptions) == 1
                            else "attivi"
                        )
                    )
                    explanation = (
                        f"La ricetta “{title_found}” è presente nel catalogo, "
                        f"ma non rispetta {active_conflicts} attualmente "
                        f"{adjective}."
                    )
                else:
                    explanation = (
                        f"La ricetta “{title_found}” è presente nel catalogo, "
                        "ma non rispetta la combinazione dei filtri attivi."
                    )

                buttons = []
                if any(kind == "preferenza" for kind, _ in conflicts):
                    buttons.append({
                        "title": "Rimuovi la preferenza",
                        "payload": "/imposta_preferenza",
                    })
                buttons.append({
                    "title": "🔄 Nuova ricerca",
                    "payload": "/nuova_ricerca",
                })
                dispatcher.utter_message(
                    text=explanation,
                    buttons=buttons,
                )
                return state_events + [SlotSet("ricerca_in_attesa", False)]

            dispatcher.utter_message(
                text="Ho trovato queste ricette con il titolo richiesto:"
            )
            proposed_ids = []
            for _, row in title_results.iterrows():
                recipe_id = str(row['original_id'])
                proposed_ids.append(recipe_id)
                payload = "/inizia_cucinare" + json.dumps(
                    {"titolo_ricetta": recipe_id}, ensure_ascii=False
                )
                message = (
                    f"🍽️ {row['Titolo']}\n"
                    f"⏱️ {row['Tempo']} | 📊 {row['Difficolta']}\n"
                    "🔗 Leggi la ricetta originale:\n"
                    f"{row['Link_Originale']}"
                )
                dispatcher.utter_message(
                    text=message,
                    buttons=[{
                        "title": "👨‍🍳 Cucina Ora (Step-by-Step)",
                        "payload": payload,
                    }],
                )

            return state_events + [
                SlotSet("ricerca_in_attesa", False),
                SlotSet("ultime_ricette_proposte", proposed_ids),
            ]

        preference_conflicts = self._preference_ingredient_conflicts(
            dietary_preferences, ingredienti
        )
        if preference_conflicts:
            dispatcher.utter_message(
                text=self._preference_conflict_message(preference_conflicts),
                buttons=self._preference_conflict_buttons(
                    multiple_preferences=len(dietary_preferences) > 1
                ),
            )
            return state_events + [SlotSet("ricerca_in_attesa", False)]

        if generic_to_clarify:
            dispatcher.utter_message(
                text=(
                    f"Quando dici {generic_to_clarify}, a cosa ti riferisci "
                    "esattamente? Scegli un'opzione oppure scrivimelo."
                ),
                buttons=self._clarification_buttons(generic_to_clarify),
            )
            return state_events + [SlotSet("ricerca_in_attesa", False)]

        if not query and latest_intent in {
            "escludi_ingrediente",
            "imposta_modalita",
            "imposta_preferenza",
        }:
            dispatcher.utter_message(
                text="Ora dimmi quali ingredienti hai a disposizione."
            )
            return state_events + [SlotSet("ricerca_in_attesa", True)]

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

                dispatcher.utter_message(text=f"Ok, cerchiamo un {tipologia}. Ma con quali ingredienti? (es: 'con {ex_ing}')")
            else:
                 dispatcher.utter_message(
                     text=(
                         "Dimmi quali ingredienti hai a disposizione, ad "
                         "esempio: “ho uova e farina”."
                     )
                 )
            
            return state_events + [SlotSet("ricerca_in_attesa", True)]

        # --- RICERCA REALE ---
        msg = "Vediamo cosa riesco a trovare"
        if tipologia: msg += f" tipo {tipologia}"
        if difficolta: msg += f", difficoltà {difficolta}"
        if tempo_req: msg += f", preparazione {tempo_req}"
        if search_mode != "guidata": msg += f", modalità {search_mode}"
        if dietary_preferences:
            preference_label = (
                "preferenza"
                if len(dietary_preferences) == 1
                else "preferenze"
            )
            msg += (
                f", {preference_label} "
                f"{self._natural_list(dietary_preferences)}"
            )
        if excluded_ingredients:
            msg += f", senza {self._natural_list(excluded_ingredients)}"
        msg += f" con: {query}..."
        dispatcher.utter_message(text=msg)

        results = engine.search(
            query,
            tipologia_filter=tipologia,
            difficolta_filter=difficolta,
            tempo_filter=tempo_req,
            excluded_ingredients=excluded_ingredients,
            dietary_preferences=dietary_preferences,
            top_n=20,
            user_ingredients=ingredienti,
        )

        if results.empty:
            requested = self._natural_list(ingredienti)
            dispatcher.utter_message(
                text=(
                    f"Con {requested} non sono riuscito a trovare una ricetta "
                    "abbastanza coerente nel mio ricettario. Se hai un altro "
                    "ingrediente, dimmelo e riprovo; altrimenti possiamo "
                    "cercare usandone soltanto uno."
                ),
                buttons=self._single_ingredient_buttons(ingredienti),
            )
            return state_events + [SlotSet("ricerca_in_attesa", False)]

        exact_results = results[results['Ingredient_Coverage'].ge(1.0)]
        if exact_results.empty:
            if search_mode == "libera":
                display_results = results.head(3)
                dispatcher.utter_message(
                    text=(
                        "Non ho trovato una corrispondenza completa, ma in "
                        "modalità libera posso mostrarti le alternative più "
                        "vicine:"
                    )
                )
            else:
                best_result = results.iloc[0]
                matched = self._natural_list(best_result['Matched_Ingredients'])
                unmatched = self._natural_list(
                    best_result['Unmatched_User_Ingredients']
                )
                requested = self._natural_list(ingredienti)

                explanation = (
                    f"Non ho trovato una ricetta che utilizzi insieme {requested}. "
                )
                if matched and unmatched:
                    explanation += (
                        f"Le proposte più vicine usano {matched}, ma non "
                        f"{unmatched}. "
                    )
                explanation += (
                    "Preferisco non mostrarti ricette poco pertinenti. Puoi "
                    "aggiungere un altro ingrediente oppure passare alla "
                    "modalità libera."
                )
                dispatcher.utter_message(
                    text=explanation,
                    buttons=self._mode_buttons(),
                )
                return state_events + [SlotSet("ricerca_in_attesa", False)]
        else:
            if search_mode == "rigorosa":
                eligible_results = exact_results[
                    exact_results['Missing_Count'].eq(0)
                ]
            elif search_mode == "flessibile":
                eligible_results = exact_results[
                    exact_results['Missing_Count'].le(2)
                ]
            else:
                eligible_results = exact_results

            if eligible_results.empty:
                limit_description = (
                    "senza ingredienti aggiuntivi"
                    if search_mode == "rigorosa"
                    else "con al massimo due ingredienti aggiuntivi"
                )
                dispatcher.utter_message(
                    text=(
                        f"Non ho trovato ricette {limit_description}. "
                        "Puoi cambiare modalità oppure aggiungere altri "
                        "ingredienti disponibili."
                    ),
                    buttons=self._mode_buttons(),
                )
                return state_events + [SlotSet("ricerca_in_attesa", False)]

            display_results = eligible_results.head(3)
            dispatcher.utter_message(
                text=(
                    "Ho trovato queste ricette, che utilizzano gli ingredienti "
                    "indicati e rispettano i vincoli impostati:"
                )
            )
        
        for index, row in display_results.iterrows():
            titolo = row['Titolo']
            ricetta_id = str(row['original_id']) 
            
            tempo = row['Tempo']
            difficolta = row['Difficolta']
            link = row['Link_Originale']
            mancanti = row['Missing_List']
            match_real = row['Matched_Real']
            unmatched_user = row['Unmatched_User_Ingredients']
            
            if len(mancanti) == 0:
                msg_extra = "✅ Hai tutto!"
            elif len(mancanti) <= 2:
                msg_extra = f"⚠️ Ti manca solo: {', '.join(mancanti)}"
            else:
                msg_extra = f"🛒 Ti mancano {len(mancanti)} ingredienti (es. {', '.join(mancanti[:3])}...)"
            
            buttons = [
                {"title": "👨‍🍳 Cucina Ora (Step-by-Step)", "payload": f"/inizia_cucinare{{\"titolo_ricetta\": \"{ricetta_id}\"}}"}
            ]

            if unmatched_user:
                match_explanation = (
                    f"🎯 Usa {match_real} ingredienti richiesti, ma non "
                    f"{self._natural_list(unmatched_user)}."
                )
            else:
                if match_real == 1:
                    match_explanation = "🎯 Usa l'ingrediente indicato."
                else:
                    match_explanation = (
                        f"🎯 Usa tutti i tuoi {match_real} ingredienti."
                    )

            messaggio = (
                f"🍽️ {titolo}\n"
                f"⏱️ {tempo} | 📊 {difficolta}\n"
                f"{match_explanation}\n"
                f"{msg_extra}\n"
                "🔗 Leggi la ricetta originale:\n"
                f"{link}"
            )
            dispatcher.utter_message(text=messaggio, buttons=buttons)

        next_suggested_ingredient = self._suggest_missing_ingredient(
            display_results
        )
        proposed_ids = [
            str(value) for value in display_results['original_id'].tolist()
        ]
        final_events = state_events + [
            SlotSet("ricerca_in_attesa", False),
            SlotSet("ultime_ricette_proposte", proposed_ids),
        ]
        if next_suggested_ingredient:
            payload = "/fornisce_ingredienti" + json.dumps(
                {"ingrediente": next_suggested_ingredient}, ensure_ascii=False
            )
            dispatcher.utter_message(
                text=(
                    f"Per restringere ancora la scelta: hai anche "
                    f"{next_suggested_ingredient}? Compare in più di una delle "
                    "ricette compatibili."
                ),
                buttons=[
                    {
                        "title": f"Sì, ho {next_suggested_ingredient}",
                        "payload": payload,
                    },
                    {
                        "title": "No, non ce l'ho",
                        "payload": "/nega_ingrediente",
                    },
                ],
            )
            final_events.append(
                SlotSet("ingrediente_suggerito", next_suggested_ingredient)
            )

        return final_events

class ActionIniziaCucinare(Action):
    def name(self) -> Text:
        return "action_inizia_cucinare"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        input_value = next(
            tracker.get_latest_entity_values("titolo_ricetta"), None
        )

        # Il classificatore puo interpretare una richiesta come avvio della
        # preparazione senza estrarre l'entita. Prima di usare le ricette
        # proposte, controlliamo quindi anche il testo nel catalogo completo.
        if not input_value:
            input_value = engine.resolve_title(
                tracker.latest_message.get("text", "")
            )

        if not input_value:
            proposed_recipes = tracker.get_slot("ultime_ricette_proposte") or []
            if isinstance(proposed_recipes, str):
                proposed_recipes = [proposed_recipes]
            proposed_recipes = [str(value) for value in proposed_recipes]

            if len(proposed_recipes) == 1:
                input_value = proposed_recipes[0]
            elif len(proposed_recipes) > 1:
                buttons = []
                for recipe_id in proposed_recipes:
                    matching = engine.df[
                        engine.df['original_id'] == int(recipe_id)
                    ] if recipe_id.isdigit() else pd.DataFrame()
                    if matching.empty:
                        continue
                    title = matching.iloc[0]['Titolo']
                    payload = "/inizia_cucinare" + json.dumps(
                        {"titolo_ricetta": recipe_id}, ensure_ascii=False
                    )
                    buttons.append({"title": title, "payload": payload})

                dispatcher.utter_message(
                    text="Quale delle ricette proposte vuoi preparare?",
                    buttons=buttons,
                )
                return []

        if not input_value:
            dispatcher.utter_message(
                text=(
                    "Prima scegliamo una ricetta. Dimmi quali ingredienti hai "
                    "oppure chiedimi una ricetta specifica."
                )
            )
            return []
        
        ricetta = pd.DataFrame()
        # Supporto Ibrido: ID Numerico (Bottone) o Titolo (Testo)
        if str(input_value).isdigit():
            rec_id = int(input_value)
            ricetta = engine.df[engine.df['original_id'] == rec_id]
        else:
            ricetta = engine.search_by_title(input_value, top_n=1)
        
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
        msg = f"👨‍🍳 Iniziamo a cucinare: {titolo_reale}\n\n🔹 Step 1/{len(step_list)}\n{primo_step}"
        
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
            dispatcher.utter_message(text=f"Missione compiuta! 🎉\nLa ricetta {titolo} è pronta per essere gustata.\nBuon appetito! 😋")
            
            return reset_search_context()
            
        next_step = step_list[next_index]
        msg = f"🔹 Step {next_index + 1}/{len(step_list)}\n{next_step}"
        
        buttons = [{"title": "Fatto! ✅ (Avanti)", "payload": "/step_successivo"}]
        
        dispatcher.utter_message(text=msg, buttons=buttons)
        
        return [SlotSet("step_corrente", next_index)]


class ActionNuovaRicerca(Action):
    def name(self) -> Text:
        return "action_nuova_ricerca"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Pulisce il contesto prima del messaggio iniziale."""
        latest_intent = tracker.latest_message.get("intent", {}).get("name", "")
        if latest_intent == "nuova_ricerca":
            dispatcher.utter_message(
                text="Ho azzerato i criteri della ricerca precedente."
            )
        return reset_search_context()


class ActionSmartFallback(Action):
    def name(self) -> Text:
        return "action_smart_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """Recupera richieste per titolo sfuggite al classificatore NLU."""
        user_text = tracker.latest_message.get("text", "")
        resolved_title = engine.resolve_title(user_text, allow_fuzzy=True)
        if resolved_title:
            dispatcher.utter_message(
                text=(
                    f"Ho riconosciuto il titolo {resolved_title}. "
                    "Cerco la ricetta nel catalogo."
                )
            )
            return [
                SlotSet("titolo_ricetta", resolved_title),
                FollowupAction("action_cerca_ricette"),
            ]

        dispatcher.utter_message(response="utter_default")
        return []

class ActionSorprendimi(Action):
    def name(self) -> Text:
        return "action_sorprendimi"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        if engine.df.empty:
            dispatcher.utter_message(
                text="Il dataset delle ricette non è disponibile."
            )
            return []

        excluded_ingredients = tracker.get_slot("ingredienti_esclusi") or []
        if isinstance(excluded_ingredients, str):
            excluded_ingredients = [excluded_ingredients]

        dietary_preferences = tracker.get_slot("preferenze_alimentari") or []
        if isinstance(dietary_preferences, str):
            dietary_preferences = [dietary_preferences]

        stored_ingredients = tracker.get_slot("ingredienti_memoria") or []
        if isinstance(stored_ingredients, str):
            stored_ingredients = [stored_ingredients]

        tipologia = tracker.get_slot("tipologia") or ""
        difficolta = tracker.get_slot("difficolta") or ""
        tempo_req = tracker.get_slot("tempo_req") or ""
        search_mode = tracker.get_slot("modalita_ricerca") or "guidata"

        if stored_ingredients:
            candidates = engine.search(
                " ".join(stored_ingredients),
                tipologia_filter=tipologia,
                difficolta_filter=difficolta,
                tempo_filter=tempo_req,
                excluded_ingredients=excluded_ingredients,
                dietary_preferences=dietary_preferences,
                top_n=len(engine.df),
                user_ingredients=stored_ingredients,
            )
            if search_mode != "libera":
                candidates = candidates[
                    candidates['Ingredient_Coverage'].ge(1.0)
                ]
            if search_mode == "rigorosa":
                candidates = candidates[candidates['Missing_Count'].eq(0)]
            elif search_mode == "flessibile":
                candidates = candidates[candidates['Missing_Count'].le(2)]
        else:
            candidates = engine.compatible_recipes(
                tipologia_filter=tipologia,
                difficolta_filter=difficolta,
                tempo_filter=tempo_req,
                excluded_ingredients=excluded_ingredients,
                dietary_preferences=dietary_preferences,
            )

        if candidates.empty:
            dispatcher.utter_message(
                text=(
                    "Non trovo una proposta casuale che rispetti tutti i "
                    "vincoli attuali. Prova a rimuoverne uno oppure indicami "
                    "altri ingredienti."
                )
            )
            return [SlotSet("ultime_ricette_proposte", None)]

        random_row = candidates.sample(n=1).iloc[0]
        
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

        dispatcher.utter_message(text="🎲 Oggi mi sento ispirato! Che ne dici di questa ricetta?")

        messaggio = (
            f"🍽️ {titolo}\n"
            f"⏱️ {tempo} | 📊 {difficolta}\n"
            "🛒 Controlla di avere gli ingredienti!\n"
            "🔗 Leggi la ricetta originale:\n"
            f"{link}"
        )
        dispatcher.utter_message(text=messaggio, buttons=buttons)

        return [
            SlotSet("titolo_ricetta", None),
            SlotSet("ultime_ricette_proposte", [ricetta_id]),
        ]
