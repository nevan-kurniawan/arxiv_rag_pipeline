import config.paths as paths
from clients.vecdb_client import VectorDBClient
from schemas.document import ArxivDocument
from utils.jsonl_utils import load_jsonl
from dotenv import load_dotenv


def main():
    load_dotenv()
    print("Rebuilding collection (delete + recreate + upsert)...")
    points = load_jsonl(ArxivDocument, paths.RAW_DATA / "arxiv_data_cache.jsonl")
    qd_client = VectorDBClient()
    try:
        qd_client.delete_collection()
    except Exception as e:
        print(f"No existing collection to delete: {e}")
    qd_client.make_collection()
    qd_client.upsert_points(points)
    print(f"Upserted {len(points)} documents.")


if __name__ == "__main__":
    main()
