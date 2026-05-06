import os
import time
from tqdm import tqdm
from clients.llm_client import LLMClient
from datetime import datetime, timezone
import json
import config
from schemas.document import ArxivDocument, SearchSyntheticGroundTruth
from utils.jsonl_utils import load_jsonl
from pathlib import Path

def build_prompt(data:ArxivDocument):
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
    prompt = prompt_template.format(**data.model_dump(mode='json'))
    return prompt

def generate_data(data: list[ArxivDocument], llm_client: LLMClient, output_path: Path):
    processed_ids = set()

    # if os.path.exists(output_path):
    #     df = pd.read_csv(output_path)
    #     counts = df['entry_id'].value_counts()
    #     valid_ids = counts[counts == 5].index
    #     df_cleaned = df[df['entry_id'].isin(valid_ids)]
    #     df_cleaned.to_csv(output_path, index=False)

    # if os.path.exists(output_path):
    #     with open(output_path, 'r', encoding='utf-8') as file:
            

    # if os.path.exists(output_path):
    #     existing_df = pd.read_csv(output_path)
    #     processed_ids = set(existing_df['entry_id'])
    
    # for entry in tqdm(data):
    #     if entry['entry_id'] in processed_ids:
    #         continue
    #     time.sleep(0.5)
    #     generated_at = datetime.now(timezone.utc)
    #     prompt = build_prompt(entry)
    #     response = llm_client.prompt_llm(prompt)
        
    #     try:
    #         json_response = response.choices[0].message.content
    #         json_response = json_response.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    #         raw_questions = json.loads(json_response)
    #         current_batch = [(question, entry['entry_id'], generated_at) for question in raw_questions]
    #         temp_df = pd.DataFrame(current_batch, columns=['question', 'entry_id', 'generated_at'])
    #         file_exists = os.path.exists(output_path)
    #         temp_df.to_csv(output_path, index=False, mode='a', header=not file_exists)

    #     except json.JSONDecodeError as e:
    #         print(f'Error parsing JSON! {e.msg} at line {e.lineno}, column {e.colno}. Skipped.')

    #     except TypeError as e:
    #         print(f"Invalid input type: Expected string, bytes, or bytearray. Received {type(json_response).__name__}. Skipped.")

    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        record = SearchSyntheticGroundTruth(**json.loads(line))
                        processed_ids.add(record.entry_id)
                    except json.JSONDecodeError:
                        continue
    
    with open(output_path, 'a', encoding='utf-8') as file:
        for entry in tqdm(data):
            if entry.entry_id in processed_ids:
                continue

            generated_at = datetime.now(timezone.utc)
            prompt = build_prompt(entry)
            response = llm_client.prompt_llm(prompt)

            try:
                content = response['response'].choices[0].message.content
                if content is None:
                    print(f"Warning: Received None content for {entry.entry_id}. Skipped.")
                    continue

                json_response = content.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
                raw_questions = json.loads(json_response)

                if not isinstance(raw_questions, list):
                    raise TypeError(f"Expected a list, received {type(raw_questions).__name__}")
                
                output_record = SearchSyntheticGroundTruth(
                    entry_id=entry.entry_id,
                    question=raw_questions,
                    question_generated_at=generated_at,
                    question_generated_by=response['model']
                )
            
                file.write(output_record.model_dump_json() + '\n')
                file.flush()

            except json.JSONDecodeError as e:
                print(f'Error parsing JSON for {entry.entry_id}: {e.msg} at line {e.lineno}, column {e.colno}. Skipped.')
            except Exception as e:
                print(f"Unexpected error processing {entry.entry_id}: {e}. Skipped.")

def main():
    from dotenv import load_dotenv
    load_dotenv()
    llm_client = LLMClient(os.environ["GROQ_API_KEY"])

    data = load_jsonl(ArxivDocument, config.RAW_DATA / 'arxiv_data_cache.jsonl')
    generate_data(data=data, llm_client=llm_client, output_path=config.GROUND_TRUTH_DIR / 'search-ground-truth-data.jsonl')

if __name__ == '__main__':
    main()