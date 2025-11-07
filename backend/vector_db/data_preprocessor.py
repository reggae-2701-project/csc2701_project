import re
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer


class DataPreprocessor:
    def __init__(self, file_path, model_name="all-MiniLM-L6-v2", vector_db_url="http://172.31.41.249:6333"):
        self.file_path = file_path
        self.vector_db_url = vector_db_url
        self.model = SentenceTransformer(model_name)

        self.hand_book_txt = self.read_text_file()
        self.hand_book_text_chunks = self.split_by_headers()
        self.hand_book_embeddings = self.create_embeddings()
        self.upload_to_vector_db()

    def read_text_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_by_headers(self) -> List[Dict[str, str]]:

        text = self.hand_book_txt.replace('\r', '')

        header_pattern = re.compile(
            r'(?P<header>^[A-Z0-9 ,\-&/\(\)]+(?:\n|$))', re.MULTILINE
        )

        parts = re.split(header_pattern, text)

        chunks = []
        current_header = None

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.isupper() or re.match(r'^[A-Z0-9 ,\-&/\(\)]+$', part):
                current_header = part
            else:
                if current_header:
                    clean_text = re.sub(r'\n{2,}', '\n', part).strip()
                    chunks.append({
                        "title": current_header,
                        "content": clean_text
                    })
                    current_header = None
                else:
                    chunks.append({
                        "title": "PREFACE",
                        "content": part
                    })
        return chunks

    def create_embeddings(self):
        embeddings = []
        for chunk in self.hand_book_text_chunks:
            vector = self.model.encode(chunk['content'])
            embeddings.append({
                "title": chunk["title"],
                "content": chunk["content"],
                "embedding": vector
            })
        return embeddings
    

    def upload_to_vector_db(self, collection_name: str="csc2701"):
        client = QdrantClient(self.vector_db_url)

        all_collections = client.get_collections().collections
        names = [d.name for d in all_collections]

        if collection_name not in names:
            print(f"Creating collection '{collection_name}'")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "mscac-dense-vector": VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    ),
                },
            )

            client.create_payload_index(
                collection_name=collection_name,
                field_name="header",
                field_schema="keyword",
            )

        else:
            print(f"Collection '{collection_name}' already exists")

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector={"mscac-dense-vector": item["embedding"].tolist()},
                    payload={
                        "header": item["title"],
                        "content": item["content"],
                        "document_title": self.file_path.split("/")[-1]
                    }
                )
                for idx, item in enumerate(self.hand_book_embeddings)
            ]
        )


if __name__ == "__main__":
    hand_book_path = "../data/docs/handbook.txt"
    data_preprocessor = DataPreprocessor(hand_book_path)
