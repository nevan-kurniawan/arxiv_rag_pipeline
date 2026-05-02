import pandas as pd
import os
import time
from tqdm import tqdm
from clients.llm_client import LLMClient
from datetime import datetime, timezone
import json
import config


def build_prompt(data:dict):
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
    prompt = prompt_template.format(**data)
    return prompt

def generate_data(data, llm_client, output_path = config.GROUND_TRUTH_DIR / 'ground-truth-data.csv'):
    processed_ids = set()

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        counts = df['entry_id'].value_counts()
        valid_ids = counts[counts == 5].index
        df_cleaned = df[df['entry_id'].isin(valid_ids)]
        df_cleaned.to_csv(output_path, index=False)

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df['entry_id'])

    # if os.path.exists(config.GROUND_TRUTH_DIR / 'failed-data.csv'):
    #     existing_failed_df = pd.read_csv(config.GROUND_TRUTH_DIR / 'failed-data.csv')
    #     processed_ids.update(existing_failed_df['entry_id'])
    
    for entry in tqdm(data):
        if entry['entry_id'] in processed_ids:
            continue
        time.sleep(0.5)
        generated_at = datetime.now(timezone.utc)
        prompt = build_prompt(entry)
        response = llm_client.prompt_llm(prompt)
        
        try:
            json_response = response.choices[0].message.content
            json_response = json_response.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            raw_questions = json.loads(json_response)
            current_batch = [(question, entry['entry_id'], generated_at) for question in raw_questions]
            temp_df = pd.DataFrame(current_batch, columns=['question', 'entry_id', 'generated_at'])
            file_exists = os.path.exists(output_path)
            temp_df.to_csv(output_path, index=False, mode='a', header=not file_exists)

        except json.JSONDecodeError as e:
            print(f'Error parsing JSON! {e.msg} at line {e.lineno}, column {e.colno}. Skipped.')
            # err_batch = [(json_response, entry['entry_id'], generated_at, e)]
            # err_df = pd.DataFrame(err_batch, columns=['raw_response', 'entry_id', 'generated_at', 'exception'])
            # failed_exists = os.path.exists(config.GROUND_TRUTH_DIR / 'failed-data.csv')
            # err_df.to_csv(config.GROUND_TRUTH_DIR / 'failed-data.csv', index=False, mode='a', header=not failed_exists)

        except TypeError as e:
            print(f"Invalid input type: Expected string, bytes, or bytearray. Received {type(json_response).__name__}. Skipped.")
            # err_batch = [(json_response, entry['entry_id'], generated_at, e)]
            # err_df = pd.DataFrame(err_batch, columns=['raw_response', 'entry_id', 'generated_at', 'exception'])
            # failed_exists = os.path.exists(config.GROUND_TRUTH_DIR / 'failed-data.csv')
            # err_df.to_csv(config.GROUND_TRUTH_DIR / 'failed-data.csv', index=False, mode='a', header=not failed_exists)

def main():
    from dotenv import load_dotenv
    load_dotenv()
    llm_client = LLMClient(os.environ["GEMINI_API_KEY"])

    with open(config.RAW_DATA / 'arxiv_data_cache.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    generate_data(data, llm_client)

if __name__ == '__main__':
    main()