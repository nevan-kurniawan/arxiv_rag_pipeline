import arxiv
from datetime import datetime, timedelta, timezone
import config.paths as paths
from typing import List
from schemas.document import ArxivDocument
from pathlib import Path


def generate_date_query(category: str = "cs.AI", days_back: int = 30) -> str:
    """Constructs the Lucene query string for ArXiv based on category and timeframe.

    Args:
        category (str, optional): Arxiv category to ingest the documents from. Defaults to 'cs.AI'.
        days_back (int, optional): Number of days to look back from today for paper submissions. Defaults to 30.

    Returns:
        str: The completed query string.
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    return f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"


def fetch_arxiv_results(
    query_string: str, max_results: int = 1000
) -> List[arxiv.Result]:
    """Initializes the ArXiv client and executes the search query.

    Args:
        query_string (str): The Lucene query string.
        max_results (int, optional): Maximum documents to ingest. Defaults to 1000.

    Returns:
        List[arxiv.Result]: List of Arxiv result object.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query_string,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def format_results(results: List[arxiv.Result]) -> List[ArxivDocument]:
    """Transforms ArXiv result objects into ArxivDocument Pydantic models

    Args:
        results (List[arxiv.Result]): List of Arxiv result object.

    Returns:
        List[ArxivDocument]: List of ArxivDocument Pydantic model.
    """
    points = []
    for result in results:
        points.append(
            ArxivDocument(
                title=result.title,
                categories=result.categories,
                authors=[author.name for author in result.authors],
                summary=result.summary,
                entry_id=result.entry_id,
                published=result.published,
            )
        )

    return points


def save_data(data: List[ArxivDocument], filepath: Path) -> None:
    """Writes the structured data to a local JSON file.

    Args:
        data (List[ArxivDocument]): list of data with ArxivDocument schema.
        filepath (Path): Path to save the results in.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as file:
        for doc in data:
            json_str = doc.model_dump_json()
            file.write(json_str + "\n")


def main(
    target_category: str = "cs.AI",
    lookback_days: int = 30,
    limit: int = 1000,
    output_path: Path = paths.RAW_DATA / "arxiv_data_cache.jsonl",
) -> None:
    """Orchestrates the data extraction, transformation, and loading process.

    Args:
        target_category (str, optional): Arxiv category to ingest the documents from. Defaults to "cs.AI".
        lookback_days (int, optional): Number of days to look back from today for paper submissions. Defaults to 30.
        limit (int, optional): Maximum documents to ingest. Defaults to 1000.
        output_path (Path, optional): Path to save the results in. Defaults to paths.RAW_DATA/"arxiv_data_cache.jsonl".
    """
    query = generate_date_query(category=target_category, days_back=lookback_days)
    raw_results = fetch_arxiv_results(query_string=query, max_results=limit)
    structured_data = format_results(raw_results)
    save_data(data=structured_data, filepath=output_path)


if __name__ == "__main__":
    main()
