from pipeline.rag import build_prompt
from clients.llm_client import LLMClient
from clients.vecdb_client import VectorDBClient
import pandas as pd
import os
from dotenv import load_dotenv
import json
import config
from tqdm import tqdm
from utils.jsonl_utils import load_jsonl
from schemas.document import SearchSyntheticGroundTruth, ResponseSyntheticGroundTruth
from pathlib import Path
import time

# def generate_responses(data: pd.DataFrame, path:str, llm_client: LLMClient, vecdb_client:VectorDBClient) -> list[dict]:
#     # TODO: Add error handling, retry logic, wait between requests, etc. The full suite.
#     results = []
#     for item in tqdm(data.to_dict(orient='records')):
#         question = item['question']
#         retrieval = vecdb_client.search(question)
#         prompt = build_prompt(retrieval, question)
#         response = llm_client.prompt_llm(prompt)

#         payloads = [point.payload for point in retrieval.points]
        
#         results.append({
#             **item,
#             'retrieved_context': payloads,
#             'response': response.choices[0].message.content
#         })
        
#     return results

def generate_responses(data: list[SearchSyntheticGroundTruth], output_path: Path, llm_client:LLMClient, vecdb_client:VectorDBClient):
    processed_ids = set()
    
    # 1. Read existing state to avoid duplicate work
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = ResponseSyntheticGroundTruth(**json.loads(line))
                        processed_ids.add(record.entry_id)
                    except json.JSONDecodeError:
                        continue
    
    # 2. Open file in append mode
    with open(output_path, 'a', encoding='utf-8') as f:
        for entry in tqdm(data, desc="Generating Responses"):
            question = entry.question[0]
            
            # Skip if already processed
            if entry.entry_id in processed_ids:
                continue
            

            try:
                retrieval = vecdb_client.search(question)
                prompt = build_prompt(retrieval, question)
                response = llm_client.prompt_llm(prompt)

                payloads = [point.payload for point in retrieval.points]
                
                result_item = {
                    **entry.model_dump(mode='json'),
                    'retrieved_context': payloads,
                    'response': response['response'].choices[0].message.content
                }
                
                f.write(json.dumps(result_item) + '\n')
                f.flush()
                
            except Exception as e:
                print(f"\nError processing question: '{question}'\nException: {e}")
                continue

    # results = []
    # with open(output_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         results.append(ResponseSyntheticGroundTruth(**json.loads(line)))
            
    # return results

def build_faithfulness_prompt(entry: ResponseSyntheticGroundTruth) -> str:
    """
    Constructs an evaluation prompt to measure how faithfully the generated 
    response adheres to the retrieved contexts.
    """
    retrieved_contexts = entry.retrieved_context
    
    # Extract and format the contexts to maximize legibility for the judging LLM
    formatted_contexts = []
    for i, ctx in enumerate(retrieved_contexts):
        title = ctx.get("title", "No Title")
        summary = ctx.get("summary", "No Summary")
        formatted_contexts.append(f"--- Context {i+1} ---\nTitle: {title}\nSummary: {summary}")
    
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
    """
    Constructs an evaluation prompt to measure how relevant the generated 
    response is to the original user question.
    """
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

def run_judges(eval_data: list[ResponseSyntheticGroundTruth], llm_client:LLMClient, output_path:Path):
    processed_ids = set()
    # 1. Read existing state to avoid duplicate work
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = ResponseSyntheticGroundTruth(**json.loads(line))
                        processed_ids.add(record.entry_id)
                    except json.JSONDecodeError:
                        continue

    with open(output_path, 'a', encoding='utf-8') as f:
        for entry in tqdm(eval_data, desc="Running Judges"):
            question = entry.question[0]
            
            if entry.entry_id in processed_ids:
                continue

            try:
                faithfulness_prompt = build_faithfulness_prompt(entry)
                relevance_prompt = build_relevance_prompt(entry)
                
                faithfulness_eval = llm_client.prompt_llm(faithfulness_prompt)
                relevance_eval = llm_client.prompt_llm(relevance_prompt)
                
                f_content = faithfulness_eval['response'].choices[0].message.content
                f_content = f_content.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
                r_content = relevance_eval['response'].choices[0].message.content
                r_content = r_content.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
                f_dict = json.loads(f_content)
                r_dict = json.loads(r_content)
            
                combined_eval = {
                    **entry.model_dump(mode='json'),
                    **f_dict,
                    **r_dict,
                    'evaluation_generated_by': relevance_eval['model']               
                }
                f.write(json.dumps(combined_eval) + '\n')
                f.flush()
            
            except Exception as e:
                print(f"\nError judging question: '{question}'\nException: {e}")
                continue

def main(path:Path = config.GROUND_TRUTH_DIR / 'response-evaluation-subset.jsonl'):
    load_dotenv()
    llm_client = LLMClient(os.environ["GROQ_API_KEY"])
    vecdb_client = VectorDBClient()
    data = load_jsonl(SearchSyntheticGroundTruth, path)
    print("Generating responses...")
    generate_responses(data, config.EVAL_DIR / 'generated_responses.jsonl', llm_client, vecdb_client)
    generated_data = load_jsonl(ResponseSyntheticGroundTruth, config.EVAL_DIR / 'generated_responses.jsonl')
    print("Running judges...")
    judged_path = config.EVAL_DIR / 'response_evaluation_results.jsonl'

    run_judges(generated_data, llm_client, judged_path)

if __name__ == "__main__":
    main()