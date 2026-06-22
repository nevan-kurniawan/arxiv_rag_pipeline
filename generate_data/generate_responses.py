from pipeline.rag import build_prompt
from clients.llm_client import LLMClient
from clients.vecdb_client import VectorDBClient
import os
from dotenv import load_dotenv
import json
import config.paths as paths
from tqdm import tqdm
from utils.jsonl_utils import load_jsonl
from schemas.document import SearchSyntheticGroundTruth, ResponseSyntheticGroundTruth
from pathlib import Path


def generate_responses(
    data: list[SearchSyntheticGroundTruth],
    output_path: Path,
    llm_client: LLMClient,
    vecdb_client: VectorDBClient,
    llm_model: str,
):
    """Run the RAG pipeline against each ground-truth question, outputting responses with retrieved context."""
    processed_ids: set[str] = set()

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = ResponseSyntheticGroundTruth(**json.loads(line))
                        processed_ids.add(record.entry_id)
                    except json.JSONDecodeError:
                        continue

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for entry in tqdm(data, desc="Generating Responses"):
            if entry.entry_id in processed_ids:
                continue

            question = entry.question[0]

            try:
                retrieval = vecdb_client.search(question, mode="rrf")
                prompt = build_prompt(retrieval, question)
                response = llm_client.prompt_llm(prompt, llm_model=llm_model)
                payloads = [point.payload for point in retrieval.points]

                result_item = {
                    **entry.model_dump(mode="json"),
                    "retrieved_context": payloads,
                    "response": response.response,
                    "response_generated_by": response.model,
                }

                f.write(json.dumps(result_item) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nError processing question: '{question}'\nException: {e}")
                continue


def main(
    subset_path: Path = paths.GROUND_TRUTH_DIR / "response-evaluation-subset.jsonl",
    generation_model: str = "llama-3.3-70b-versatile",
):
    load_dotenv()
    llm_client = LLMClient(
        provider="groq",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    vecdb_client = VectorDBClient()
    data = load_jsonl(SearchSyntheticGroundTruth, subset_path)

    generated_path = paths.EVAL_DIR / "generated_responses.jsonl"

    print("Generating responses...")
    generate_responses(
        data, generated_path, llm_client, vecdb_client, llm_model=generation_model
    )


if __name__ == "__main__":
    main()
