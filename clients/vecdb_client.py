from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from schemas.document import ArxivDocument
import uuid
import os
from typing import Literal


class VectorDBClient:
    """_summary_"""

    def __init__(
        self,
        dense_embedding_model_handle: str = "jinaai/jina-embeddings-v2-small-en",
        sparse_embedding_model_handle: str = "Qdrant/bm25",
    ):
        """_summary_

        Args:
            dense_embedding_model_handle (str, optional): _description_. Defaults to "jinaai/jina-embeddings-v2-small-en".
            sparse_embedding_model_handle (str, optional): _description_. Defaults to 'Qdrant/bm25'.
        """
        self.client = QdrantClient(
            api_key=os.environ["QDRANT_API_KEY"],
            url=os.environ["QDRANT_CLUSTER_ENDPOINT"],
        )

        # self.collection_list = self.client.get_collections()
        self.dense_embedder = TextEmbedding(model_name=dense_embedding_model_handle)
        self.sparse_embedder = SparseTextEmbedding(
            model_name=sparse_embedding_model_handle
        )

    def make_collection(self, collection_name: str = "arxiv-rag"):
        """_summary_

        Args:
            collection_name (str, optional): _description_. Defaults to 'arxiv-rag'.
        """
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_embedder.embedding_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            print(f"Collection {collection_name} created!")
        else:
            print(f"Collection {collection_name} already exists!")

    def delete_collection(self, collection_name: str = "arxiv-rag"):
        """_summary_

        Args:
            collection_name (str, optional): _description_. Defaults to 'arxiv-rag'.
        """
        self.client.delete_collection(collection_name=collection_name)
        print(f"Collection {collection_name} deleted!")

    def upsert_points(
        self, docs: list[ArxivDocument], collection_name: str = "arxiv-rag"
    ):
        """_summary_

        Args:
            docs (list[ArxivDocument]): _description_
            collection_name (str, optional): _description_. Defaults to 'arxiv-rag'.
        """

        docs_list = list(docs)
        raw_text = [doc.summary for doc in docs_list]
        dense_embedding = list(self.dense_embedder.embed(raw_text))
        sparse_embedding = list(self.sparse_embedder.embed(raw_text))

        qdrant_points = []

        for i, doc in enumerate(docs_list):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc.summary))
            payload_dict = doc.model_dump(mode="json")

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embedding[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_embedding[i].indices.tolist(),
                            values=sparse_embedding[i].values.tolist(),
                        ),
                    },
                    payload=payload_dict,
                )
            )

        self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def search(
        self,
        query: str,
        mode:Literal["dense", "sparse", "cascade", "rrf"],
        collection_name: str = "arxiv-rag",
        prefetch_limit: int = 100,
        top_k: int = 10,
    ):
        dense_query_embedding = next(iter(self.dense_embedder.embed([query]))).tolist()
        sparse_query_embedding = next(iter(self.sparse_embedder.embed([query])))
        if mode == "cascade":
            result = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query_embedding.indices.tolist(),
                            values=sparse_query_embedding.values.tolist(),
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                    )
                ],
                query=dense_query_embedding,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
        elif mode == "dense":
            result = self.client.query_points(
                collection_name=collection_name,
                query=dense_query_embedding,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
        elif mode == "sparse":
            result = self.client.query_points(
                query=models.SparseVector(
                    indices=sparse_query_embedding.indices.tolist(),
                    values=sparse_query_embedding.values.tolist(),
                ),
                using="sparse",
                limit=top_k,
                collection_name=collection_name,
                with_payload=True,
            )
        elif mode == "rrf":
            result = self.client.query_points(
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query_embedding.indices.tolist(),
                            values=sparse_query_embedding.values.tolist(),
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=dense_query_embedding,
                        using="dense",
                        limit=prefetch_limit,
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                collection_name=collection_name,
                with_payload=True,
            )
        else:
            raise ValueError

        return result