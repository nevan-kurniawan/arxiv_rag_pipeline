from pathlib import Path
import json
from collections.abc import Callable

def load_jsonl[T](model: Callable[..., T], file_path: Path) -> list[T]:
    """Reads a JSONL file and returns a list of entries instantiated as the provided model.

    Returns:
        _type_: _description_
    """
    
    entries: list[T] = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entries.append(model(**json.loads(line)))
            
    return entries

# def write_jsonl(data: list[dict], file_path: Path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         for item in data:
#             file.write(json.dumps(item) + '\n')

# def append_jsonl(data: list[dict], file_path: Path):
#     with open(file_path, 'a', encoding='utf-8') as file:
#         for item in data:
#             file.write(json.dumps(item) + '\n')