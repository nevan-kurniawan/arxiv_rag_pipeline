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
    """Run the RAG pipeline against each ground-truth question, persisting responses with retrieved context."""
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
                retrieval = vecdb_client.search(question)
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


def build_faithfulness_prompt(entry: ResponseSyntheticGroundTruth) -> str:
    """Build the LLM-as-judge prompt measuring whether the response is grounded in the retrieved context."""
    retrieved_contexts = entry.retrieved_context

    formatted_contexts = []
    for i, ctx in enumerate(retrieved_contexts):
        title = ctx.get("title", "No Title")
        summary = ctx.get("summary", "No Summary")
        formatted_contexts.append(f"--- Context {i + 1} ---\nTitle: {title}\nSummary: {summary}")

    contexts_text = "\n\n".join(formatted_contexts)
    response = entry.response

    prompt = f"""You are an objective evaluator measuring the 'faithfulness' of a generated response.

Faithfulness evaluates whether the response is strictly grounded in the provided context on a scale of 1 to 3:
- 1 (Low): The response contains information entirely unsupported by, or contradictory to, the retrieved context.
- 2 (Medium): The response is partially supported but includes some hallucinations or unverified external claims.
- 3 (High): The response is fully supported by the retrieved context without introducing external information.

### Retrieved Contexts:
{contexts_text}

### Generated Response:
{response}

Please analyze the content and context of the generated answer in relation to the retrieved context and provide your evaluation in parsable JSON without using code blocks:
{{ "explanation_faithfulness": "[Provide a brief explanation for your evaluation]", "faithfulness": 1 | 2 | 3}}""".strip()

    return prompt


def build_relevance_prompt(entry: ResponseSyntheticGroundTruth) -> str:
    """Build the LLM-as-judge prompt measuring whether the response directly answers the question."""
    question = entry.question[0]
    response = entry.response

    prompt = f"""You are an objective evaluator measuring the 'response relevance' of a generated response.

Response relevance evaluates how well the generated response directly answers the provided question on a scale of 1 to 3:
- 1 (Low): The response is completely irrelevant to the core intent of the question.
- 2 (Medium): The response partially addresses the question but misses the core intent or includes significant tangential information.
- 3 (High): The response directly, clearly, and comprehensively answers the question.

### Question:
{question}

### Generated Response:
{response}

Please analyze the content and context of the generated answer in relation to the question and provide your evaluation in parsable JSON without using code blocks:
{{ "explanation_relevance": "[Provide a brief explanation for your evaluation]", "relevance": 1 | 2 | 3}}""".strip()

    return prompt


def run_judges(
    eval_data: list[ResponseSyntheticGroundTruth],
    llm_client: LLMClient,
    llm_model: str,
    output_path: Path,
):
    """Run faithfulness and relevance judges over generated responses, persisting parsed scores."""
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
        for entry in tqdm(eval_data, desc="Running Judges"):
            if entry.entry_id in processed_ids:
                continue

            question = entry.question[0]

            try:
                faithfulness_prompt = build_faithfulness_prompt(entry)
                relevance_prompt = build_relevance_prompt(entry)

                faithfulness_eval = llm_client.prompt_llm(faithfulness_prompt, llm_model=llm_model)
                relevance_eval = llm_client.prompt_llm(relevance_prompt, llm_model=llm_model)

                f_content = (
                    faithfulness_eval.response.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                r_content = (
                    relevance_eval.response.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )

                f_dict = json.loads(f_content)
                r_dict = json.loads(r_content)

                combined_eval = {
                    **entry.model_dump(mode="json"),
                    **f_dict,
                    **r_dict,
                    "evaluation_generated_by": relevance_eval.model,
                }
                f.write(json.dumps(combined_eval) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nError judging question: '{question}'\nException: {e}")
                continue


def main(
    subset_path: Path = paths.GROUND_TRUTH_DIR / "response-evaluation-subset.jsonl",
    generation_model: str = "llama-3.3-70b-versatile",
    judge_model: str = "openai/gpt-oss-120b",
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
    judged_path = paths.EVAL_DIR / "response_evaluation_results.jsonl"

    print("Generating responses...")
    generate_responses(data, generated_path, llm_client, vecdb_client, llm_model=generation_model)

    generated_data = load_jsonl(ResponseSyntheticGroundTruth, generated_path)

    print("Running judges...")
    run_judges(generated_data, llm_client, judge_model, judged_path)


if __name__ == "__main__":
    main()