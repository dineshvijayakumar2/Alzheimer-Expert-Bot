from langchain.embeddings.base import Embeddings
from openai import AzureOpenAI
from typing import List

class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, openai_key: str, azure_endpoint: str,
                 api_version: str = "2023-05-15",
                 deployment_name: str = "text-embedding-3-small"):
        self.client = AzureOpenAI(
            api_key=openai_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment_name
            )
            embeddings.extend([data.embedding for data in response.data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.deployment_name
        )
        return response.data[0].embedding

