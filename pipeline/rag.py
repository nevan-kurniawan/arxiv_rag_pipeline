from clients.llm_client import LLMClient
from clients.vecdb_client import VectorDBClient


def build_prompt(query_results, query) -> str:
    system_prompt = (
        "You are a chatbot designed to synthesize and summarize the results of a search "
        "on the Arxiv database of papers in the last 30 days based on the user's queries. "
        f"QUERY: {query}\n\n"
        "Based on the query, the search returned these documents that might be relevant. Review the following retrieved documents carefully:\n\n"
    )

    formatted_results = []
    iter_len = len(query_results.points)
    for i in range(0, iter_len):
        # Dynamically handle an arbitrary number of authors using .join()
        authors_list = query_results.points[i].payload.get(
            "authors", ["Unknown Author"]
        )
        authors_str = ", ".join(authors_list)
        categories_list = query_results.points[i].payload.get(
            "categories", ["Unknown category"]
        )
        categories_str = ", ".join(categories_list)
        title = query_results.points[i].payload.get("title", "Untitled")

        # Properly utilize f-strings for variable interpolation
        result_block = (
            f"RESULT {i + 1}:\n"
            f"    Title: {title}\n"
            f"    Authors: {authors_str}\n"
            f"    Categories: {categories_str}\n"
            f"    Abstract: {query_results.points[i].payload.get('summary', ['Unknown abstract'])}\n"
            f"    Published date: {query_results.points[i].payload.get('published', ['Unknown date'])}\n"
            f"    Entry ID: {query_results.points[i].payload.get('entry_id', ['Unknown ID'])}\n"
        )
        formatted_results.append(result_block)

    # Assemble the final prompt efficiently
    final_prompt = system_prompt + "\n".join(formatted_results)
    final_prompt_add = final_prompt + "\nNow, answer the user's query."
    return final_prompt_add.strip()


def query(
    question: str,
    vecdb_client: VectorDBClient,
    llm_client: LLMClient,
    llm_model,
    limit: int = 3,
) -> str:
    retrieval = vecdb_client.search(question, limit=limit)
    prompt = build_prompt(retrieval, question)
    response = llm_client.prompt_llm(prompt, llm_model=llm_model)
    reply = response.response
    if reply is not None:
        return reply
    else:
        return "Reply error"
