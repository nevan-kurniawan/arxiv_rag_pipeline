import os
from tqdm import tqdm
from clients.llm_client import LLMClient
from datetime import datetime, timezone
import json
import config.paths as paths
from schemas.document import ArxivDocument, SearchSyntheticGroundTruth
from utils.jsonl_utils import load_jsonl
from pathlib import Path


def build_prompt(data: ArxivDocument) -> str:
    """Build the prompt that asks an LLM to generate 5 retrieval questions for one paper."""
    prompt_template = """
    Your task is to generate a synthetic dataset for an Arxiv retrieval pipeline. For every entry, formulate 5 questions a user might ask where the entry would naturally be retrieved as a source of information to assist in answering the user's questions. The entry should contain relevant information to the questions, and the questions should be complete and not too short. If possible, use as few words as possible from the entry. Formulate questions using your own words rather than phrases from the entry.

    The entry:
    title: {title}
    categories: {categories}
    authors: {authors}
    summary: {summary}
    entry_id: {entry_id}
    published: {published}

    Provide the output in parsable JSON without using code blocks:
    ["question1", "question2", ..., "question5"]
    """.strip()
    return prompt_template.format(**data.model_dump(mode="json"))


def generate_data(
    data: list[ArxivDocument],
    llm_client: LLMClient,
    llm_model: str,
    output_path: Path,
):
    """Generate 5 synthetic retrieval questions per paper, with idempotent resume."""
    processed_ids: set[str] = set()

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    try:
                        record = SearchSyntheticGroundTruth(**json.loads(line))
                        processed_ids.add(record.entry_id)
                    except json.JSONDecodeError:
                        continue

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as file:
        for entry in tqdm(data):
            if entry.entry_id in processed_ids:
                continue

            generated_at = datetime.now(timezone.utc)
            prompt = build_prompt(entry)

            try:
                response = llm_client.prompt_llm(prompt, llm_model=llm_model)
                content = response.response

                if not content:
                    print(f"Warning: empty content for {entry.entry_id}. Skipped.")
                    continue

                json_response = (
                    content.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                raw_questions = json.loads(json_response)

                if not isinstance(raw_questions, list):
                    raise TypeError(f"Expected a list, received {type(raw_questions).__name__}")

                output_record = SearchSyntheticGroundTruth(
                    entry_id=entry.entry_id,
                    question=raw_questions,
                    question_generated_at=generated_at,
                    question_generated_by=response.model,
                )

                file.write(output_record.model_dump_json() + "\n")
                file.flush()

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {entry.entry_id}: {e.msg} at line {e.lineno}, column {e.colno}. Skipped.")
            except Exception as e:
                print(f"Unexpected error processing {entry.entry_id}: {e}. Skipped.")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    llm_client = LLMClient(
        provider="groq",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    data = load_jsonl(ArxivDocument, paths.RAW_DATA / "arxiv_data_cache.jsonl")
    generate_data(
        data=data,
        llm_client=llm_client,
        llm_model="llama-3.3-70b-versatile",
        output_path=paths.GROUND_TRUTH_DIR / "search-ground-truth-data.jsonl",
    )


if __name__ == "__main__":
    main()