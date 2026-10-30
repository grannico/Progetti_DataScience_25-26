"""Valida e riallinea tipologie e lookup con il dataset delle ricette.

La tipologia è associata stabilmente al titolo, quindi un riordinamento del
CSV non cambia la classificazione. Per una nuova ricetta bisogna prima
aggiungere manualmente la relativa voce a ``tipologie_ricette.json``.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_DIR / "actions" / "dataset_ricette_giallozafferano.csv"
TYPE_FILE = PROJECT_DIR / "actions" / "tipologie_ricette.json"
LOOKUP_FILE = PROJECT_DIR / "data" / "titoli_lookup.yml"


def build_title_mapping(records: Iterable[dict]) -> Dict[str, str]:
    """Costruisce la mappa titolo-tipologia rifiutando duplicati o vuoti."""
    mapping: Dict[str, str] = {}
    for record in records:
        title = str(record.get("Titolo", "")).strip()
        category = str(record.get("Tipologia", "")).strip()
        if not title or not category:
            raise ValueError("Ogni metadato deve avere Titolo e Tipologia")
        if title in mapping:
            raise ValueError(f"Titolo duplicato nei metadati: {title}")
        mapping[title] = category
    return mapping


def align_metadata(dataset_titles: List[str], mapping: Dict[str, str]) -> List[dict]:
    """Riordina i metadati per titolo e verifica la corrispondenza completa."""
    if len(dataset_titles) != len(set(dataset_titles)):
        raise ValueError("Il dataset contiene titoli duplicati")

    dataset_set = set(dataset_titles)
    mapped_set = set(mapping)
    missing = sorted(dataset_set - mapped_set)
    obsolete = sorted(mapped_set - dataset_set)
    if missing or obsolete:
        raise ValueError(
            "Metadati non allineati al dataset. "
            f"Titoli mancanti={missing}; titoli non presenti={obsolete}"
        )

    return [
        {"Titolo": title, "Tipologia": mapping[title]}
        for title in dataset_titles
    ]


def main() -> None:
    recipes = pd.read_csv(SOURCE_FILE)
    dataset_titles = recipes["Titolo"].astype(str).str.strip().tolist()
    existing_records = json.loads(TYPE_FILE.read_text(encoding="utf-8"))
    metadata = align_metadata(
        dataset_titles,
        build_title_mapping(existing_records),
    )

    TYPE_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lookup_lines = [
        'version: "3.1"',
        "",
        "nlu:",
        "- lookup: titolo_ricetta",
        "  examples: |",
    ]
    lookup_lines.extend(f"    - {title}" for title in dataset_titles)
    LOOKUP_FILE.write_text("\n".join(lookup_lines) + "\n", encoding="utf-8")

    print(f"Verificate {len(metadata)} tipologie in {TYPE_FILE}")
    print(f"Creati {len(metadata)} titoli in {LOOKUP_FILE}")


if __name__ == "__main__":
    main()
