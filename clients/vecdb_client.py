from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

class VectorDBClient:
    def __init__(self, port_link: str = "http://localhost:6333", embedding_model_handle: str):
        self.client = QdrantClient(port_link)
        self.collection = self.client.get_collections()
        self.embedder = TextEmbedding(model_name=embedding_model_handle)

    def upsert_points(self, points, model_handle):
        id = 0
        qdrant_points = []

        for point in points:
            entry = models.PointStruct(
                id=id,
                vector=models.Document(text=point['summary'], model=model_handle),
                payload={
                    **point,
                    'id': id
                } #save all needed metadata fields
            )
            qdrant_points.append(entry)

            id += 1