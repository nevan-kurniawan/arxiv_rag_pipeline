import arxiv
from datetime import datetime, timedelta, timezone
import config
from typing import List
from schemas.document import ArxivDocument
from pathlib import Path

def generate_date_query(category: str = 'cs.AI', days_back: int = 30) -> str:
    """Constructs the Lucene query string for ArXiv based on category and timeframe.    

    Args:
        category (str, optional): _description_. Defaults to 'cs.AI'.
        days_back (int, optional): _description_. Defaults to 30.

    Returns:
        str: _description_
    """
    # 1. Calculate time in UTC, as explicitly required by the ArXiv API
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    # 2. Format to YYYYMMDDHHMM (ArXiv's "TTTT" is hours and minutes in 24h format)
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")
    
    return f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"

def fetch_arxiv_results(query_string: str, max_results: int = 100) -> List[arxiv.Result]:
    """Initializes the ArXiv client and executes the search query.

    Args:
        query_string (str): _description_
        max_results (int, optional): _description_. Defaults to 100.

    Returns:
        List[arxiv.Result]: _description_
    """
    # 4. Initialize client and search parameters
    client = arxiv.Client()
    search = arxiv.Search(
        query=query_string,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    return list(client.results(search))

def format_results(results: List[arxiv.Result]) -> List[ArxivDocument]:
    """Transforms ArXiv result objects into a standardized dictionary format.

    Args:
        results (List[arxiv.Result]): _description_

    Returns:
        List[ArxivDocument]: _description_
    """
    points = []
    for result in results:
        points.append(
            ArxivDocument(
                title = result.title,
                categories = result.categories,
                authors = [author.name for author in result.authors],
                summary = result.summary,
                entry_id = result.entry_id,
                published = result.published
            )
        )
    
    return points

def save_data(data: List[ArxivDocument], filepath: Path) -> None:
    """Writes the structured data to a local JSON file safely.

    Args:
        data (List[Dict[str, Any]]): _description_
        filepath (Path): _description_
    """
    filepath.parent.mkdir(parents=True, exist_ok=True) 

    with open(filepath, 'w', encoding='utf-8') as file:
        for doc in data:
            json_str = doc.model_dump_json()
            file.write(json_str + '\n')

def main(
        target_category: str = "cs.AI",
        lookback_days: int = 30,
        limit: int = 100,
        output_path: Path = config.RAW_DATA / 'arxiv_data_cache.jsonl'
) -> None:
    """Orchestrates the data extraction, transformation, and loading process.

    Args:
        target_category (str, optional): _description_. Defaults to "cs.AI".
        lookback_days (int, optional): _description_. Defaults to 30.
        limit (int, optional): _description_. Defaults to 100.
        output_path (Path, optional): _description_. Defaults to config.RAW_DATA/'arxiv_data_cache.jsonl'.
    """
    # Execution Pipeline
    query = generate_date_query(category=target_category, days_back=lookback_days)
    raw_results = fetch_arxiv_results(query_string=query, max_results=limit)
    structured_data = format_results(raw_results)
    save_data(data=structured_data, filepath=output_path)
    
if __name__ == '__main__':
    # TODO: Add Typer integration for parsing
    main()