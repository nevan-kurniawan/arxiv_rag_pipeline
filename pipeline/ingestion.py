import arxiv
from datetime import datetime, timedelta, timezone
import json
import config

def main():
    # 1. Calculate time in UTC, as explicitly required by the ArXiv API
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    # 2. Format to YYYYMMDDHHMM (ArXiv's "TTTT" is hours and minutes in 24h format)
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    # 3. Construct the exact Lucene query string
    query_string = f"cat:cs.AI AND submittedDate:[{start_str} TO {end_str}]"

    # 4. Initialize client and search parameters
    client = arxiv.Client()
    search = arxiv.Search(
        query=query_string,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = list(client.results(search))

    # Assuming 'results_list' is your materialized list from client.results(search)
    points = []
    for result in results:
        points.append({
            'title': result.title,
            'categories': result.categories, # Already a list of strings
            'authors': [author.name for author in result.authors], # Extract string names
            'summary': result.summary,
            'entry_id': result.entry_id,
            'published': result.published.isoformat() # Convert datetime to string
        })

    # Write to a local file
    with open(config.RAW_DATA / 'arxiv_data_cache.json', 'w', encoding='utf-8') as file:
        json.dump(points, file, indent=4)
    
if __name__ == '__main__':
    main()