from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

class VectorDBClient:
    def __init__(self, dense_embedding_model_handle: str = "jinaai/jina-embeddings-v2-small-en",
                 sparse_embedding_model_handle: str = 'Qdrant/bm25',
                 port_link: str = "http://localhost:6333"):
        self.client = QdrantClient(port_link)
        self.collection_list = self.client.get_collections()
        self.dense_embedder = TextEmbedding(model_name=dense_embedding_model_handle)
        self.sparse_embedder = SparseTextEmbedding(model_name=sparse_embedding_model_handle)

    def make_collection(self, collection_name:str = 'arxiv-rag'):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config= {
                    'dense': models.VectorParams(
                        size = self.dense_embedder.embedding_size,
                        distance = models.Distance.COSINE
                    )},
                sparse_vectors_config= {'sparse': models.SparseVectorParams()}
            )
            print(f'Collection {collection_name} created!')
        else:
            print(f'Collection {collection_name} already exists!')

    def delete_collection(self, collection_name:str = 'arxiv-rag'):
        self.client.delete_collection(collection_name=collection_name)
        print(f"Collection {collection_name} deleted!")

    def upsert_points(self, docs:list[dict], collection_name:str = 'arxiv-rag'):

        id = [i for i in range(0, len(docs))]
        raw_text = [doc['summary'] for doc in docs]
        dense_embedding = list(self.dense_embedder.embed(raw_text))
        sparse_embedding = list(self.sparse_embedder.embed(raw_text))
        payload = docs

        qdrant_points = [
            models.PointStruct(
                id=id[i],
                vector = {
                        'dense': list(dense_embedding[i]),
                        'sparse': models.SparseVector(
                            indices = sparse_embedding[i].indices.tolist(),
                            values = sparse_embedding[i].values.tolist()
                        )
                    },
                payload = {
                    **payload[i],
                    'id': id[i]
                    }
        ) for i in range(0, len(docs))]

        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points
        )
    
    def search(self, query:str, collection_name:str = 'arxiv-rag', limit:int = 3):
        dense_query_embedding = next(iter(self.dense_embedder.embed([query]))).tolist()
        sparse_query_embedding = next(iter(self.sparse_embedder.embed([query])))
        result = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query = models.SparseVector(
                        indices = sparse_query_embedding.indices.tolist(),
                        values = sparse_query_embedding.values.tolist(),
                    ),
                    using = 'sparse',
                    limit = limit
                )
            ],
            query = dense_query_embedding,
            using='dense',
            limit = limit,
            with_payload=True
        )

        return result