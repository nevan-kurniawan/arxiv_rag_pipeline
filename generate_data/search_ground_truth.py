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
    """Build the prompt that asks an LLM to generate 5 retrieval questions for one paper.

    Questions are written from the searcher's information need before they know the
    paper exists.
    """
    prompt_template = """
    You generate realistic retrieval questions for an Arxiv search benchmark. The goal is to simulate what a real researcher would type into a search box BEFORE they know this paper exists.

    Given one paper, write exactly 5 questions such that this paper would be a strong answer — but each question must describe a general INFORMATION NEED or PROBLEM, not this specific paper.

    Hard rules:
    - Do NOT name any method, model, system, framework, benchmark, dataset, or metric coined by this paper. Describe the general concept instead. If the paper introduces "FooNet for X", ask about "X" or "approaches to X", never "FooNet".
    - Do NOT refer to "the authors", "the study", "the paper", "this work", "the proposed method", or any framing that presupposes the reader has seen the paper.
    - Do NOT paraphrase the abstract's contribution sentences. Someone who has not read this paper should find every question natural — describing a problem, not a specific solution.
    - Phrase each question from the perspective of someone who HAS the problem but does NOT yet know the solution. Describe the problem, not the answer.
    - Keep the problem specific enough that this paper is genuinely the best answer. Keep the *task, setting, and constraint* precise; drop only the *coined name and contribution framing*. "Quantizing a state-space model to ternary weights without retraining from scratch" is specific enough. "Making models smaller" is too generic and is wrong.
    - Vary the angle across the 5: motivation, method, trade-offs, evaluation, limitations — always at the level of the general problem.

    Process (think before writing):
    1. Identify the underlying problem this paper addresses, stripped of the paper's own naming and framing.
    2. Write 5 questions a researcher facing that problem might search for, before they know this paper exists.

    Return ONLY a JSON array of exactly 5 plain strings. Each element must be a string, not an object. Do not include entry_id, relevance, keys, nesting, commentary, or markdown.

    Correct format (placeholders show shape only — write real questions about the problem):
    ["<question 1>", "<question 2>", "<question 3>", "<question 4>", "<question 5>"]

    Incorrect format (do NOT do this):
    [{{"question": "..."}}, {{"entry_id": "...", "question": "..."}}]

    Paper:
    title: {title}
    categories: {categories}
    authors: {authors}
    summary: {summary}
    """.strip()
    return prompt_template.format(**data.model_dump(mode="json"))


def generate_data(
    data: list[ArxivDocument],
    llm_client: LLMClient,
    llm_model: str,
    output_path: Path,
):
    """Generate 5 synthetic retrieval questions per paper."""
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
                    raise TypeError(
                        f"Expected a list, received {type(raw_questions).__name__}"
                    )

                output_record = SearchSyntheticGroundTruth(
                    entry_id=entry.entry_id,
                    question=raw_questions,
                    question_generated_at=generated_at,
                    question_generated_by=response.model,
                )

                file.write(output_record.model_dump_json() + "\n")
                file.flush()

            except json.JSONDecodeError as e:
                print(
                    f"Error parsing JSON for {entry.entry_id}: {e.msg} at line {e.lineno}, column {e.colno}. Skipped."
                )
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
    data = load_jsonl(
        ArxivDocument, paths.GROUND_TRUTH_DIR / "corpus_raw_subsample.jsonl"
    )
    generate_data(
        data=data,
        llm_client=llm_client,
        llm_model="qwen/qwen3-32b",
        output_path=paths.GROUND_TRUTH_DIR / "search-ground-truth-data.jsonl",
    )


if __name__ == "__main__":
    main()
