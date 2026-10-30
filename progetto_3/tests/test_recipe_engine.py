import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.actions import (
    ActionCercaRicette,
    ActionIniziaCucinare,
    ActionNuovaRicerca,
    ActionSmartFallback,
    ActionSorprendimi,
    RecipeEngine,
)
from scripts.sync_recipe_metadata import align_metadata, build_title_mapping


class RecipeEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        dataset = pd.DataFrame(
            [
                {
                    "Titolo": "Pasta con zucchine",
                    "Ingredienti": "pasta, zucchine, olio, sale",
                    "Procedimento": "Cuocere la pasta e unire le zucchine.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "20 min",
                    "Link_Originale": "https://example.com/pasta-zucchine",
                    "Tipologia": "primo",
                },
                {
                    "Titolo": "Pollo con zucchine",
                    "Ingredienti": "pollo, zucchine, cipolle, olio, sale",
                    "Procedimento": "Cuocere il pollo e aggiungere le zucchine.",
                    "Immagine": "",
                    "Difficolta": "Media",
                    "Tempo": "45 min",
                    "Link_Originale": "https://example.com/pollo-zucchine",
                    "Tipologia": "secondo",
                },
                {
                    "Titolo": "Torta al cioccolato",
                    "Ingredienti": "farina, uova, cioccolato, burro, zucchero",
                    "Procedimento": "Preparare l'impasto e cuocere la torta.",
                    "Immagine": "",
                    "Difficolta": "Difficile",
                    "Tempo": "1 h 30 min",
                    "Link_Originale": "https://example.com/torta-cioccolato",
                    "Tipologia": "dolce",
                },
                {
                    "Titolo": "Insalata di mais",
                    "Ingredienti": "mais, pomodori, olio, sale",
                    "Procedimento": "Mescolare il mais con i pomodori e condire.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "10 min",
                    "Link_Originale": "https://example.com/insalata-mais",
                    "Tipologia": "contorno",
                },
                {
                    "Titolo": "Pollo alle arachidi",
                    "Ingredienti": "pollo, arachidi, olio, sale",
                    "Procedimento": "Cuocere il pollo e aggiungere le arachidi.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "30 min",
                    "Link_Originale": "https://example.com/pollo-arachidi",
                    "Tipologia": "secondo",
                },
                {
                    "Titolo": "Spaghetti alla Carbonara",
                    "Ingredienti": "spaghetti, guanciale, tuorli, pecorino, pepe",
                    "Procedimento": "Cuocere gli spaghetti e unirli al condimento.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "25 min",
                    "Link_Originale": "https://example.com/carbonara",
                    "Tipologia": "primo",
                },
                {
                    "Titolo": "Pollo rustico",
                    "Ingredienti": (
                        "petto di pollo, cipolle dorate, pomodori pelati, "
                        "peperoncino piccante, scorza di limone, "
                        "sovracosce di pollo"
                    ),
                    "Procedimento": "Cuocere il pollo con le verdure.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "35 min",
                    "Link_Originale": "https://example.com/pollo-rustico",
                    "Tipologia": "secondo",
                },
                {
                    "Titolo": "Torta di ricotta",
                    "Ingredienti": "ricotta vaccina, uova, zucchero",
                    "Procedimento": "Mescolare la ricotta e cuocere il dolce.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "45 min",
                    "Link_Originale": "https://example.com/torta-ricotta",
                    "Tipologia": "dolce",
                },
                {
                    "Titolo": "Ravioli di ricotta",
                    "Ingredienti": "ricotta di pecora, pasta fresca, sale",
                    "Procedimento": "Farcire la pasta con la ricotta e cuocere.",
                    "Immagine": "",
                    "Difficolta": "Media",
                    "Tempo": "1 h",
                    "Link_Originale": "https://example.com/ravioli-ricotta",
                    "Tipologia": "primo",
                },
                {
                    "Titolo": "Seppie con piselli",
                    "Ingredienti": "seppie, piselli, pomodoro, olio",
                    "Procedimento": "Rosolare le seppie. Aggiungere i piselli e cuocere.",
                    "Immagine": "",
                    "Difficolta": "Facile",
                    "Tempo": "50 min",
                    "Link_Originale": "https://example.com/seppie-piselli",
                    "Tipologia": "secondo",
                },
            ]
        )
        csv_path = Path(self.temp_dir.name) / "recipes.csv"
        dataset.to_csv(csv_path, index=False)
        self.engine = RecipeEngine(csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_duration_conversion(self):
        self.assertEqual(RecipeEngine._duration_to_minutes("20 min"), 20)
        self.assertEqual(RecipeEngine._duration_to_minutes("1 h 30 min"), 90)
        self.assertIsNone(RecipeEngine._duration_to_minutes("Nota"))
        self.assertEqual(RecipeEngine._requested_max_minutes("entro 45 minuti"), 45)
        self.assertEqual(RecipeEngine._requested_max_minutes("entro 1 ora"), 60)
        self.assertEqual(RecipeEngine._requested_max_minutes("veloce"), 30)

    def test_search_uses_tfidf_similarity(self):
        results = self.engine.search("pollo zucchine", top_n=3)

        self.assertEqual(results.iloc[0]["Titolo"], "Pollo con zucchine")
        self.assertGreater(results.iloc[0]["TFIDF_Similarity"], 0)
        self.assertGreater(
            results.iloc[0]["Ranking_Score"],
            results.iloc[0]["Heuristic_Score"],
        )

    def test_search_prioritizes_all_requested_ingredients(self):
        results = self.engine.search("pollo cipolla", top_n=3)

        self.assertEqual(results.iloc[0]["Titolo"], "Pollo con zucchine")
        self.assertEqual(results.iloc[0]["Ingredient_Coverage"], 1.0)
        self.assertEqual(results.iloc[0]["Unmatched_User_Ingredients"], [])

    def test_search_marks_partial_matches(self):
        results = self.engine.search("pollo cioccolato", top_n=3)

        self.assertTrue(results["Ingredient_Coverage"].lt(1.0).all())
        self.assertIn(
            "cioccolato", results.iloc[0]["Unmatched_User_Ingredients"]
        )

    def test_search_filters_difficulty(self):
        results = self.engine.search("zucchine", difficolta_filter="facile")

        self.assertEqual(results["Titolo"].tolist(), ["Pasta con zucchine"])

    def test_search_filters_quick_recipes(self):
        results = self.engine.search("zucchine", tempo_filter="veloce")

        self.assertEqual(results["Titolo"].tolist(), ["Pasta con zucchine"])
        durations = results["Tempo"].apply(RecipeEngine._duration_to_minutes)
        self.assertTrue(durations.le(30).all())

    def test_search_excludes_unavailable_ingredient(self):
        results = self.engine.search(
            "zucchine", excluded_ingredients=["pollo"], top_n=3
        )

        self.assertNotIn("Pollo con zucchine", results["Titolo"].tolist())

    def test_vegetarian_preference_excludes_meat(self):
        results = self.engine.search(
            "zucchine", dietary_preferences=["vegetariana"], top_n=3
        )

        self.assertNotIn("Pollo con zucchine", results["Titolo"].tolist())
        self.assertIn("Pasta con zucchine", results["Titolo"].tolist())

    def test_lactose_exclusion_filters_butter(self):
        results = self.engine.search(
            "cioccolato", excluded_ingredients=["lattosio"], top_n=3
        )

        self.assertTrue(results.empty)

    def test_mais_and_arachidi_are_searchable(self):
        mais_results = self.engine.search("mais")
        arachidi_results = self.engine.search("arachidi")

        self.assertEqual(mais_results.iloc[0]["Titolo"], "Insalata di mais")
        self.assertEqual(
            arachidi_results.iloc[0]["Titolo"], "Pollo alle arachidi"
        )

    def test_arachidi_can_be_excluded(self):
        results = self.engine.search(
            "pollo", excluded_ingredients=["arachidi"], top_n=10
        )

        self.assertNotIn("Pollo alle arachidi", results["Titolo"].tolist())
        self.assertIn("Pollo con zucchine", results["Titolo"].tolist())

    def test_search_by_recipe_title(self):
        results = self.engine.search_by_title("carbonara")

        self.assertEqual(
            results.iloc[0]["Titolo"], "Spaghetti alla Carbonara"
        )

    def test_title_search_respects_dietary_preferences(self):
        results = self.engine.search_by_title(
            "carbonara", dietary_preferences=["vegana"]
        )

        self.assertTrue(results.empty)

    def test_type_filter_uses_dataset_category(self):
        results = self.engine.compatible_recipes(tipologia_filter="secondo")

        self.assertIn("Pollo con zucchine", results["Titolo"].tolist())
        self.assertNotIn("Torta al cioccolato", results["Titolo"].tolist())
        self.assertTrue(results["Tipologia"].eq("secondo").all())

    def test_missing_count_uses_structured_ingredients(self):
        results = self.engine.search(
            "pollo", top_n=10, user_ingredients=["pollo"]
        )
        row = results[results["Titolo"].eq("Pollo rustico")].iloc[0]

        self.assertEqual(row["Missing_Count"], 4)
        self.assertEqual(
            set(row["Missing_List"]),
            {"cipolla", "pomodoro", "peperoncino", "limone"},
        )
        for descriptor in {
            "petto", "dorate", "pelati", "piccante", "scorza", "sovracosce"
        }:
            self.assertNotIn(descriptor, row["Missing_List"])

    def test_generic_name_matches_structured_ingredient_variant(self):
        results = self.engine.search(
            "ricotta", top_n=20, user_ingredients=["ricotta"]
        )
        ricotta_rows = results[
            results["Titolo"].isin(["Torta di ricotta", "Ravioli di ricotta"])
        ]

        self.assertEqual(len(ricotta_rows), 2)
        self.assertTrue(ricotta_rows["Ingredient_Coverage"].eq(1.0).all())
        self.assertTrue(ricotta_rows["Matched_Real"].eq(1).all())
        self.assertTrue(
            ricotta_rows["Missing_List"].apply(
                lambda values: all("ricotta" not in value for value in values)
            ).all()
        )

    def test_color_adjectives_do_not_create_mixed_ingredient_names(self):
        for value in [
            "cipolle rosse",
            "cipolla rossa",
            "cipolle bianche",
            "cipolle nere",
            "cipolle verdi",
        ]:
            with self.subTest(value=value):
                self.assertEqual(
                    self.engine._canonical_ingredient(value), "cipolla"
                )

    def test_title_resolver_uses_full_catalogue(self):
        self.assertEqual(
            self.engine.resolve_title("come si fa la carbonara?"),
            "Spaghetti alla Carbonara",
        )

    def test_title_resolver_handles_introductory_words_and_small_typo(self):
        self.assertEqual(
            self.engine.resolve_title(
                "Dammi la ricetta per gli spaghetti alla carbonaraa"
            ),
            "Spaghetti alla Carbonara",
        )
        self.assertEqual(
            self.engine.resolve_title("Come preparo la carbonara?"),
            "Spaghetti alla Carbonara",
        )

    def test_title_resolver_does_not_force_unrelated_request(self):
        self.assertEqual(
            self.engine.resolve_title("che tempo farà domani a Roma?"), ""
        )


class RecipeMetadataTestCase(unittest.TestCase):
    def test_categories_follow_titles_after_dataset_reorder(self):
        mapping = build_title_mapping([
            {"Titolo": "Ricetta A", "Tipologia": "primo"},
            {"Titolo": "Ricetta B", "Tipologia": "dolce"},
        ])

        aligned = align_metadata(["Ricetta B", "Ricetta A"], mapping)

        self.assertEqual(aligned, [
            {"Titolo": "Ricetta B", "Tipologia": "dolce"},
            {"Titolo": "Ricetta A", "Tipologia": "primo"},
        ])

    def test_categories_reject_unknown_dataset_title(self):
        mapping = {"Ricetta A": "primo"}

        with self.assertRaises(ValueError):
            align_metadata(["Ricetta A", "Ricetta nuova"], mapping)


class ActionCercaRicetteTestCase(unittest.TestCase):
    @staticmethod
    def _tracker(slots, intent, entities, text="test"):
        return Tracker(
            sender_id="test-user",
            slots=slots,
            latest_message={
                "text": text,
                "intent": {"name": intent, "confidence": 1.0},
                "entities": entities,
            },
            events=[],
            paused=False,
            followup_action=None,
            active_loop={},
            latest_action_name=None,
        )

    @staticmethod
    def _slot_values(events):
        return {
            event["name"]: event.get("value")
            for event in events
            if event.get("event") == "slot"
        }

    def test_new_search_resets_old_filters(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ricerca_in_attesa": False,
                "tipologia": "primo",
                "difficolta": "facile",
                "tempo_req": "veloce",
            },
            intent="cerca_ricetta",
            entities=[
                {"entity": "ingrediente", "value": "uova"},
                {"entity": "ingrediente", "value": "zucchine"},
                {"entity": "ingrediente", "value": "patate"},
            ],
        )

        events = ActionCercaRicette().run(
            CollectingDispatcher(), tracker, {}
        )
        slots = self._slot_values(events)

        self.assertEqual(
            slots["ingredienti_memoria"], ["uova", "zucchine", "patate"]
        )
        self.assertIsNone(slots["tipologia"])
        self.assertIsNone(slots["difficolta"])
        self.assertIsNone(slots["tempo_req"])

    def test_additional_ingredient_is_accumulated(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["uova", "zucchine", "patate"],
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="fornisce_ingredienti",
            entities=[{"entity": "ingrediente", "value": "formaggio"}],
        )

        events = ActionCercaRicette().run(
            CollectingDispatcher(), tracker, {}
        )
        slots = self._slot_values(events)

        self.assertEqual(
            slots["ingredienti_memoria"],
            ["uova", "zucchine", "patate", "formaggio"],
        )

    def test_affirmative_reply_uses_suggested_ingredient(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["melanzane", "pomodoro"],
                "ingrediente_suggerito": "basilico",
                "ingredienti_esclusi": None,
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="fornisce_ingredienti",
            entities=[],
            text="ho anche quell'ingrediente",
        )

        events = ActionCercaRicette().run(
            CollectingDispatcher(), tracker, {}
        )
        slots = self._slot_values(events)

        self.assertEqual(
            slots["ingredienti_memoria"],
            ["melanzane", "pomodoro", "basilico"],
        )

    def test_partial_match_gets_conversational_response(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="cerca_ricetta",
            entities=[
                {"entity": "ingrediente", "value": "salmone"},
                {"entity": "ingrediente", "value": "cioccolato"},
            ],
        )
        dispatcher = CollectingDispatcher()

        ActionCercaRicette().run(dispatcher, tracker, {})

        response = dispatcher.messages[-1]
        self.assertIn("Preferisco non mostrarti", response["text"])
        self.assertEqual(len(response["buttons"]), 3)
        self.assertEqual(
            response["buttons"][-1]["title"], "Ricerca libera"
        )

    def test_generic_ingredient_asks_for_clarification(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
            },
            intent="cerca_ricetta",
            entities=[
                {"entity": "ingrediente", "value": "carne"},
                {"entity": "ingrediente", "value": "cipolla"},
            ],
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(slots["ingrediente_da_chiarire"], "carne")
        self.assertIn("a cosa ti riferisci", dispatcher.messages[-1]["text"])
        self.assertEqual(len(dispatcher.messages[-1]["buttons"]), 4)

    def test_specific_ingredient_replaces_generic_one(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["carne", "cipolla"],
                "ingrediente_da_chiarire": "carne",
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="fornisce_ingredienti",
            entities=[{"entity": "ingrediente", "value": "manzo"}],
        )

        events = ActionCercaRicette().run(
            CollectingDispatcher(), tracker, {}
        )
        slots = self._slot_values(events)

        self.assertEqual(
            slots["ingredienti_memoria"], ["cipolla", "manzo"]
        )
        self.assertIsNone(slots["ingrediente_da_chiarire"])

    def test_generic_fish_asks_for_clarification(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
            },
            intent="cerca_ricetta",
            entities=[
                {"entity": "ingrediente", "value": "pesce"},
                {"entity": "ingrediente", "value": "patate"},
            ],
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(slots["ingrediente_da_chiarire"], "pesce")
        button_titles = [
            button["title"] for button in dispatcher.messages[-1]["buttons"]
        ]
        self.assertEqual(
            button_titles, ["Salmone", "Tonno", "Merluzzo", "Orata"]
        )

    def test_denied_suggestion_is_excluded_and_does_not_surprise(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["uova"],
                "ingrediente_suggerito": "salsa",
                "ingredienti_esclusi": None,
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="nega_ingrediente",
            entities=[],
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(slots["ingredienti_esclusi"], ["salsa"])
        self.assertIn("escludo salsa", dispatcher.messages[0]["text"])
        self.assertNotIn("Oggi mi sento ispirato", dispatcher.messages[0]["text"])

    def test_denial_can_also_add_an_available_ingredient(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["patate"],
                "ingrediente_suggerito": "timo",
                "ingredienti_esclusi": None,
                "preferenze_alimentari": ["vegetariana"],
                "modalita_ricerca": "guidata",
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="nega_ingrediente",
            entities=[
                {"entity": "ingrediente_escluso", "value": "timo"},
                {"entity": "ingrediente", "value": "burro"},
            ],
            text="non ho il timo ma ho il burro",
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)
        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )

        self.assertEqual(slots["ingredienti_memoria"], ["patate", "burro"])
        self.assertEqual(slots["ingredienti_esclusi"], ["timo"])
        self.assertIn("escludo timo e aggiungo burro", response_text)
        self.assertIn("con: patate burro", response_text)
        self.assertIn("Rosti", response_text)
        self.assertIn("Hai tutto!", response_text)
        self.assertNotIn("hai anche burro", response_text.casefold())

    def test_search_mode_is_saved(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["zucchine"],
                "modalita_ricerca": "guidata",
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "ricerca_in_attesa": False,
            },
            intent="imposta_modalita",
            entities=[{
                "entity": "modalita_ricerca",
                "value": "flessibile",
            }],
        )

        dispatcher = CollectingDispatcher()
        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(slots["modalita_ricerca"], "flessibile")
        self.assertNotIn("*", dispatcher.messages[0]["text"])

    def test_dietary_preference_is_saved(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["zucchine"],
                "modalita_ricerca": "guidata",
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "ricerca_in_attesa": False,
            },
            intent="imposta_preferenza",
            entities=[{
                "entity": "preferenza_alimentare",
                "value": "vegetariano",
            }],
        )

        dispatcher = CollectingDispatcher()
        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(slots["preferenze_alimentari"], ["vegetariana"])
        self.assertIn(
            "preferenza vegetariana", dispatcher.messages[0]["text"]
        )
        self.assertNotIn(
            "preferenze vegetariana", dispatcher.messages[0]["text"]
        )

    def test_preference_and_ingredient_in_same_message_are_both_used(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "modalita_ricerca": "guidata",
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "ricerca_in_attesa": False,
            },
            intent="imposta_preferenza",
            entities=[
                {
                    "entity": "preferenza_alimentare",
                    "value": "vegetariano",
                },
                {"entity": "ingrediente", "value": "patate"},
            ],
            text="sono vegetariano, ho solo le patate",
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})
        slots = self._slot_values(events)
        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )

        self.assertEqual(slots["ingredienti_memoria"], ["patate"])
        self.assertEqual(slots["preferenze_alimentari"], ["vegetariana"])
        self.assertFalse(slots["ricerca_in_attesa"])
        self.assertNotIn("Ora dimmi quali ingredienti", response_text)
        self.assertIn("con: patate", response_text)

    def test_incompatible_preference_and_ingredient_explain_conflict(self):
        cases = [
            ("vegetariano", "carne", "preferenza vegetariana"),
            ("vegano", "uova", "preferenza vegana"),
            ("celiaco", "pasta", "vincolo senza glutine"),
            ("senza lattosio", "latte", "vincolo senza lattosio"),
        ]

        for preference, ingredient, expected_criterion in cases:
            with self.subTest(preference=preference, ingredient=ingredient):
                tracker = self._tracker(
                    slots={
                        "ingredienti_memoria": None,
                        "modalita_ricerca": "guidata",
                        "preferenze_alimentari": None,
                        "ingredienti_esclusi": None,
                        "ingrediente_da_chiarire": None,
                        "ricerca_in_attesa": False,
                    },
                    intent="imposta_preferenza",
                    entities=[
                        {
                            "entity": "preferenza_alimentare",
                            "value": preference,
                        },
                        {"entity": "ingrediente", "value": ingredient},
                    ],
                    text=f"{preference} con {ingredient}",
                )
                dispatcher = CollectingDispatcher()

                events = ActionCercaRicette().run(dispatcher, tracker, {})
                slots = self._slot_values(events)
                response = dispatcher.messages[-1]

                self.assertIn("Ho rilevato un conflitto", response["text"])
                self.assertIn(ingredient, response["text"])
                self.assertIn(expected_criterion, response["text"])
                self.assertEqual(
                    [button["title"] for button in response["buttons"]],
                    ["Rimuovi la preferenza", "🔄 Nuova ricerca"],
                )
                self.assertTrue(
                    all(
                        len(button["payload"].encode("utf-8")) <= 64
                        for button in response["buttons"]
                    )
                )
                self.assertEqual(slots["ingredienti_memoria"], [ingredient])

    def test_preference_conflict_precedes_generic_clarification(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "modalita_ricerca": "guidata",
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "ingrediente_da_chiarire": None,
                "ricerca_in_attesa": False,
            },
            intent="imposta_preferenza",
            entities=[
                {
                    "entity": "preferenza_alimentare",
                    "value": "vegetariano",
                },
                {"entity": "tipologia", "value": "secondo"},
                {"entity": "ingrediente", "value": "carne"},
            ],
            text="sono vegetariano e voglio un secondo a base di carne",
        )
        dispatcher = CollectingDispatcher()

        ActionCercaRicette().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        self.assertIn("Ho rilevato un conflitto", response_text)
        self.assertNotIn("a cosa ti riferisci", response_text)

    def test_strict_mode_without_ingredients_asks_for_them(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "modalita_ricerca": "guidata",
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "ricerca_in_attesa": False,
            },
            intent="imposta_modalita",
            entities=[{
                "entity": "modalita_ricerca",
                "value": "rigorosa",
            }],
            text="modalità rigorosa",
        )
        dispatcher = CollectingDispatcher()

        events = ActionCercaRicette().run(dispatcher, tracker, {})

        self.assertTrue(self._slot_values(events)["ricerca_in_attesa"])
        self.assertIn(
            "dimmi quali ingredienti", dispatcher.messages[-1]["text"].lower()
        )


class SelectionActionsTestCase(unittest.TestCase):
    def setUp(self):
        RecipeEngineTestCase.setUp(self)

    def tearDown(self):
        RecipeEngineTestCase.tearDown(self)

    @staticmethod
    def _tracker(slots, intent, entities=None, text="test"):
        return Tracker(
            sender_id="test-user",
            slots=slots,
            latest_message={
                "text": text,
                "intent": {"name": intent, "confidence": 1.0},
                "entities": entities or [],
            },
            events=[],
            paused=False,
            followup_action=None,
            active_loop={},
            latest_action_name=None,
        )

    @staticmethod
    def _slot_values(events):
        return {
            event["name"]: event.get("value")
            for event in events
            if event.get("event") == "slot"
        }

    def test_surprise_respects_preferences_and_exclusions(self):
        tracker = self._tracker(
            slots={
                "preferenze_alimentari": ["vegana"],
                "ingredienti_esclusi": ["mais"],
                "ingredienti_memoria": None,
                "modalita_ricerca": "guidata",
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="sorprendimi",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionSorprendimi().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        self.assertIn("Pasta con zucchine", response_text)
        self.assertNotIn("Pollo", response_text)
        self.assertNotIn("mais", response_text.casefold())
        self.assertNotIn("**", response_text)
        self.assertNotIn("[Leggi", response_text)
        self.assertIn("https://", response_text)
        self.assertEqual(
            self._slot_values(events)["ultime_ricette_proposte"], ["0"]
        )

    def test_recipe_messages_use_plain_text_and_plain_url(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ingredienti_esclusi": None,
                "preferenze_alimentari": None,
                "modalita_ricerca": "guidata",
                "ricerca_in_attesa": False,
                "tipologia": None,
                "difficolta": None,
                "tempo_req": None,
            },
            intent="cerca_ricetta",
            entities=[{"entity": "ingrediente", "value": "mais"}],
            text="ho del mais",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            ActionCercaRicette().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        self.assertNotIn("**", response_text)
        self.assertNotIn("[Leggi", response_text)
        self.assertIn("https://example.com/insalata-mais", response_text)

    def test_text_start_uses_only_proposed_recipe(self):
        tracker = self._tracker(
            slots={"ultime_ricette_proposte": ["0"]},
            intent="inizia_cucinare",
            text="cucina ora",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionIniziaCucinare().run(dispatcher, tracker, {})

        self.assertIn("Iniziamo a cucinare: Pasta con zucchine", dispatcher.messages[0]["text"])
        self.assertNotIn("**", dispatcher.messages[0]["text"])
        self.assertEqual(
            self._slot_values(events)["titolo_ricetta_corrente"],
            "Pasta con zucchine",
        )

    def test_text_start_asks_when_more_recipes_were_proposed(self):
        tracker = self._tracker(
            slots={"ultime_ricette_proposte": ["0", "3"]},
            intent="inizia_cucinare",
            text="guidami nella preparazione",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionIniziaCucinare().run(dispatcher, tracker, {})

        self.assertEqual(events, [])
        self.assertIn("Quale", dispatcher.messages[0]["text"])
        self.assertEqual(len(dispatcher.messages[0]["buttons"]), 2)
        self.assertNotIn("Errore tecnico", dispatcher.messages[0]["text"])

    def test_text_start_resolves_known_title_before_previous_results(self):
        tracker = self._tracker(
            slots={"ultime_ricette_proposte": ["0", "3"]},
            intent="inizia_cucinare",
            text="mi mostri come preparare le seppie con piselli?",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionIniziaCucinare().run(dispatcher, tracker, {})

        self.assertIn(
            "Iniziamo a cucinare: Seppie con piselli",
            dispatcher.messages[0]["text"],
        )
        self.assertEqual(
            self._slot_values(events)["titolo_ricetta_corrente"],
            "Seppie con piselli",
        )

    def test_specific_recipe_request_searches_title(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ricerca_in_attesa": False,
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "modalita_ricerca": "guidata",
            },
            intent="cerca_ricetta",
            entities=[{
                "entity": "titolo_ricetta",
                "value": "carbonara",
            }],
            text="come si fa la carbonara?",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionCercaRicette().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        self.assertIn("Spaghetti alla Carbonara", response_text)
        self.assertEqual(
            self._slot_values(events)["ultime_ricette_proposte"], ["5"]
        )

    def test_extracted_typo_title_is_compared_with_catalogue(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ricerca_in_attesa": False,
                "preferenze_alimentari": None,
                "ingredienti_esclusi": None,
                "modalita_ricerca": "guidata",
            },
            intent="cerca_ricetta_titolo",
            entities=[{
                "entity": "titolo_ricetta",
                "value": "spaghetti alla carbonaraa",
            }],
            text="Dammi la ricetta per gli spaghetti alla carbonaraa",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionCercaRicette().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        self.assertIn("Spaghetti alla Carbonara", response_text)
        self.assertEqual(
            self._slot_values(events)["ultime_ricette_proposte"], ["5"]
        )

    def test_known_title_explains_active_preference_conflict(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["patate", "zucchine"],
                "ricerca_in_attesa": False,
                "preferenze_alimentari": ["vegetariana"],
                "ingredienti_esclusi": None,
                "modalita_ricerca": "libera",
            },
            intent="cerca_ricetta_titolo",
            entities=[{
                "entity": "titolo_ricetta",
                "value": "carbonara",
            }],
            text="Come preparo la carbonara?",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            ActionCercaRicette().run(dispatcher, tracker, {})

        response = dispatcher.messages[-1]
        self.assertEqual(
            response["text"],
            "La ricetta “Spaghetti alla Carbonara” è presente nel catalogo, "
            "ma non rispetta la preferenza vegetariana attualmente attiva.",
        )
        self.assertEqual(
            [button["title"] for button in response["buttons"]],
            ["Rimuovi la preferenza", "🔄 Nuova ricerca"],
        )
        self.assertTrue(
            all(
                len(button["payload"].encode("utf-8")) <= 64
                for button in response["buttons"]
            )
        )

    def test_remove_preference_button_retries_blocked_title(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": None,
                "ricerca_in_attesa": False,
                "preferenze_alimentari": ["vegetariana"],
                "ingredienti_esclusi": None,
                "modalita_ricerca": "guidata",
                "titolo_ricetta": "Spaghetti alla Carbonara",
            },
            intent="imposta_preferenza",
            entities=[],
            text="/imposta_preferenza",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionCercaRicette().run(dispatcher, tracker, {})

        response_text = " ".join(
            message.get("text", "") for message in dispatcher.messages
        )
        slots = self._slot_values(events)
        self.assertIsNone(slots["preferenze_alimentari"])
        self.assertIn("Spaghetti alla Carbonara", response_text)
        self.assertIn("Ho trovato queste ricette", response_text)

    def test_new_search_resets_every_slot_and_restores_defaults(self):
        tracker = self._tracker(
            slots={
                "ingrediente": ["patate", "zucchine"],
                "ingredienti_memoria": ["patate", "zucchine"],
                "ricerca_in_attesa": True,
                "ingrediente_da_chiarire": "carne",
                "ingrediente_suggerito": "cipolla",
                "ingredienti_esclusi": ["aglio"],
                "preferenze_alimentari": ["vegetariana"],
                "modalita_ricerca": "libera",
                "tipologia": "primo",
                "difficolta": "facile",
                "tempo_req": "veloce",
                "titolo_ricetta": "Carbonara",
                "titolo_ricetta_corrente": "Carbonara",
                "ultime_ricette_proposte": ["5"],
                "lista_step": ["Passaggio"],
                "step_corrente": 2,
            },
            intent="nuova_ricerca",
            text="nuova ricerca",
        )
        dispatcher = CollectingDispatcher()

        events = ActionNuovaRicerca().run(dispatcher, tracker, {})
        slots = self._slot_values(events)

        self.assertEqual(events[0]["event"], "reset_slots")
        self.assertEqual(slots["modalita_ricerca"], "guidata")
        self.assertFalse(slots["ricerca_in_attesa"])
        self.assertEqual(slots["step_corrente"], 0)
        self.assertIn("azzerato", dispatcher.messages[0]["text"])

    def test_start_resets_existing_context_without_confirmation(self):
        tracker = self._tracker(
            slots={
                "ingredienti_memoria": ["patate"],
                "preferenze_alimentari": ["vegetariana"],
                "modalita_ricerca": "libera",
            },
            intent="greet",
            text="/start",
        )
        dispatcher = CollectingDispatcher()

        events = ActionNuovaRicerca().run(dispatcher, tracker, {})

        self.assertEqual(events[0]["event"], "reset_slots")
        self.assertEqual(self._slot_values(events)["modalita_ricerca"], "guidata")
        self.assertEqual(dispatcher.messages, [])

    def test_fallback_recovers_known_recipe_title(self):
        tracker = self._tracker(
            slots={},
            intent="nlu_fallback",
            text="vorrei cucinare la carbonara",
        )
        dispatcher = CollectingDispatcher()

        with patch("actions.actions.engine", self.engine):
            events = ActionSmartFallback().run(dispatcher, tracker, {})

        slots = self._slot_values(events)
        self.assertEqual(slots["titolo_ricetta"], "Spaghetti alla Carbonara")
        self.assertEqual(events[-1]["name"], "action_cerca_ricette")
        self.assertIn("Spaghetti alla Carbonara", dispatcher.messages[0]["text"])


if __name__ == "__main__":
    unittest.main()
