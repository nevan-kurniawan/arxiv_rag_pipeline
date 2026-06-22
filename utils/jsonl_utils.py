from pathlib import Path
import json
from collections.abc import Callable


def load_jsonl[T](model: Callable[..., T], file_path: Path) -> list[T]:
    """Reads a JSONL file and returns a list of entries instantiated as the provided model."""

    entries: list[T] = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entries.append(model(**json.loads(line)))

    return entries
