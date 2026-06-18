Schemas:
    class ArxivDocument(BaseModel):
        """_summary_

        Args:
            BaseModel (_type_): _description_
        """
        title: str
        categories: list[str]
        authors: list[str]
        summary: str
        entry_id: str
        published: datetime

    class SearchSyntheticGroundTruth(BaseModel):
        entry_id: str
        question: list[str]
        question_generated_at: datetime
        question_generated_by: str

    class ResponseSyntheticGroundTruth(SearchSyntheticGroundTruth):
        retrieved_context: list[dict]
        response: str
        response_generated_by: str

    class TokenDetails(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int

    class LLMResponse(BaseModel):
        response: str
        provider: str
        model: str
        token_details: TokenDetails
        generated_at: datetime

Clients:
    VectorDBClient:
        __init__:
            Initializes the client. 
            Args:
                - dense_embedding_model_handle (default jinaai/jina-embeddings-v2-small-en)
                - sparse_embedding_model_handle (default Qdrant/bm25)
            
            Attributes:
                - client: A QdrantClient object with port 6333, api_key from env, url endpoint from env
                - dense embedder: a TextEmbedding object
                - sparse embedder: a SparseTextEmbedding object
        
        make_collection:
            input: takes collection_name, defaults to arxiv_rag
            process:
            - Checks if collection exists or not
            - If not, creates collection with collection_name and dense vector config being size of embedding_size, and distance = cosine distance. As well as sparse vector config taken from models.SparseVectorParams
            - Else, outputs that collection already exists
        
        delete_collection:
            deletes collection
        
        upsert_points:
            input:
                - docs: list[ArxivDocument] to be upserted
                - collection_name
            
            process:
                - docs_list = list(docs)
                - takes doc.summary for every document in docs_list, gives raw_text = list[str]
                - dense_embedding
                - sparse_embedding
                - create empty qdrant_points list
                - for every i, entry in enumerate(docs_list):
                    point_id = string of doc.summary turned into uuid5 hash using the url namespace

                    payload_dict = dump the doc into json dictionary



                    qdrant_points.append(
                        models.pointstruct(
                            id = point id,
                            vector = {
                                dense vector = dense_embedding[i].tolist() 
                                # Note: dense_embedding is list of numpyArray by the way, likely the actual raw vectors so it's turned to a list,
                                sparse vector = models.SparseVector(indices = sparse_embedding[i].indices.tolist(),
                                values = sparse_embedding[i].values.tolist()
                                # Note: sparse_embedding is a list of sparse_embedding, indices is an NDArray (not sure what that is), and values is the actual embedding (NumpyArray)
                            },
                            payload = payload_dict (the actual entry to be retrieved. So vector is the thing used to query, and id is the unique identifier (UUID5))
                        )
                    )
                
                - upsert the qdrant points into the collection
        search
            input: query string, collection name, limit for results (defaults to 3)
            process:
                dense_query_embedding = turn string to dense query embedding with dense embedder
                sparse_query_embedding = turn string to sparse query embedding with sparse embedder
                result = query points(
                    collection name,
                    prefetch = [
                        models.Prefetch(
                            query = models.SparseVector(
                                indices=sparse_query_embedding.indices.tolist(),
                                values=sparse_query_embedding.values.tolist(),
                            ),
                            using = 'sparse'
                            limit = limit,
                        )
                    ],
                    # Note: prefetch gives the first pass result using the above setting
                    query = dense query embedding
                    using dense
                    limit = limit
                    with payload
                )
                return output
            output: search result (of type QueryResponse)
    LLMClient:
        __init__:
            provider string, api key , base url (these two are passed from the app currently)
            stores provider string and llm_client object which is an OpenAI object
        
        prompt_llm()
            input: prompt string, model string
            process:
            get response from llm
            get token_details with schema TokenDetails

            pass token details to the final schema in the return
            return LLMResponse object

Flow:
    ingestion:
        1. ingestion.py
            Input: 
                Defined in file: target_category, lookback_days, limit, output_path
            Process:
                - Constructs the Lucene query string for ArXiv based on category and timeframe (generate_date_query)
                - Initializes the ArXiv client and executes the search query (fetch_arxiv_results)
                - Transforms ArXiv result objects into a standardized dictionary format. (format_results)
                - save the data to a local Jsonl file (save_data)
            Output:
                JSONL with each entry being an ArxivDocument object. Saved in 'data/raw/arxiv_data_cache.jsonl'

        2. arxiv_data_cache.jsonl
        3. vecdb_client.py
        4. qdrant cloud (used to be control_panel.ipynb, but I should make a cloud upsertion script later)
    
    Serving:
        app.py:
            1. User enters the website
            2. Streamlit app spins up (app.py)
            3. app.py flow:
                open "config/model_configs.yaml"
                provider_options at sidebar, list of config['providers'].keys()
                #Note: options are groq and google
                selected_provider at the sidebar
                essentially just prompts the user for various settings (provider, model), then grabs my own API key for connecting to the providers (I should make this inputtable later)
                instantiates vecdb and llm clients. llm client instantiated with the selected provider, api key, and the base url (taken from the model config yaml)

                user inputs question

                uses query method from pipeline/rag.py, taking in the question, vecdb client, and llm client, and selected model from the llm client's selected provider
                write the answer
        rag.py:
            1. build_prompt(query_results, query)
                Input:
                    query_results QueryResponse object from vecdb search
                    query: actual query string
                process:
                    various unpacking and formatting operations for creating the query for the LLM
                Output:
                    final prompt
            2. query
                Input:
                    question, actual query string
                    vecdb client passed from the app
                    llm client passed from the app
                    llm model string to use for that provider's model selection
                    limit of returned queries used for the retrieval from vecdb
                process:
                    search from vecdb
                    builds prompt from limit amount of retrieved documents
                    response using llm_client's prompt_llm function
                    reply is the response object's response.
                output:
                    the reply string from the llm response

    Evaluation:
        search_ground_truth.py:
            build prompt:
                Input:
                    a single ArxivDocument object
                Process:
                    make prompt (for every entry, formulate 5 retrieval questions plausible)
                Output:
                    prompt

            generate_data:
                Input:
                    list of ArxivDocument object
                    LLMClient object
                    llm_model string
                    output path
                Process:
                    check if output path exists
                        implements idempotency by loading each line and if it's a valid output(validated against the SearchSyntheticGroundTruth object), adds the entry id to already processed id
                    for every entry, check if exists, build prompt, get response, check existence, formatting, load the response into json. If invalid, raise error
                    If valid, then load into the SearchSyntheticGroundTruth model. Then, write to file then flush

            main:
                instantiate client
                load raw data
                generate search ground truth
        
        response_evaluation_subset_generation.py
            load search ground truth
            to dataframe (I think I should learn polars soon. But for next project. Out of scope right now.)
            unique papers df random sampling one question per paper, we have full set but each has one question
            grab question
            check len
            sample just 100 out of these (more than 100)
            export to json
        
        search_evaluation.py:
            I'll review each metric later. But everything works fine.
            run_evaluation:
                input:
                    list of SearchSyntheticGroundTruth
                    search funciton takes vectordb object
                    top k takes how many results get returned
                
                process:
                    instantiate list
                    for every entry in the searhc ground truth json, 
                        get the entry id
                        for every question in each entry (5):
                            search in the vecdb for top k returns,
                            check matches by checking if the entry id in payload equals the entry id
                            each matrix returned could be of a different length than top k, so we pad by multiplying false to the length of limit - length of match, and append to any matrix that doesn't conform to the shape to ensure the vectorized numpy operation works.
                            append the match arr of the question to bool match list
                    
                    convert list to np array
                    return evaluation metrics

                output:
                    eval results

            main:
                instantiate db, ground truth load, result run evaluation
                export
            
    
    response_evaluation.py:
        generate_response:
            input:
                data search synthetic ground truth
                output path
                llm client
                vec db client
                llm model string
            
            process:
                create empty setof processed ids
                idempotency check same like before
                open output path
                for every entry in data:
                    idempotency check
                    question is the first item in the question attribute since it's a list dir
                    search vecdb
                    build rag prompt
                    get response
                    get payload
            output:
                response synthetic ground truth object
        
        build_faithfulness_prompt:
            input:
                a single response synthetic ground truth object
            process:
                builds a faithfulness prompt, asking to rank from 1 to 3
            
            output:
                faithfulness prompt
        
        build relevance prompt:
            input:
                a single response synthetic ground truth object
            process:
                builds a relevance prompt, asking to rank from 1 to 3
            
            output:
                relevance prompt
        
        run_judges:
            input:
                list of all reponse synthetic ground truth object
                llm client (judge model)
                llm model (judge model)
                output path
            
            process:
                idempotency set
                idempotency check
                run faithfulness and relevance llm as judge
                get the responses
                clean the responses and validate if it's a valid json
                write output to file (I think I should make another object here to validate the data contract for the output of the entire llm as judge eval pipeline?)
            output:
                finished llm as judge faithfulness and relevance eval file