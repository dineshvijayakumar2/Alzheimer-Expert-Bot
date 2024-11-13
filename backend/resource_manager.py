# resource_manager.py
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, HttpUrl
from langchain.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document
from pymongo import MongoClient
import hashlib
from datetime import datetime
from config import config
import logging
import requests
from urllib.parse import urlparse
import os
from text_cleaning import TextCleaner, preprocess_documents
import PyPDF2


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom PDF loader implementation
class CustomPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                'source': self.file_path,
                                'page': page_num + 1,
                                'total_pages': len(pdf_reader.pages)
                            }
                        ))
        except Exception as e:
            logging.error(f"Error reading PDF {self.file_path}: {str(e)}")
            raise
        return documents




class Resource(BaseModel):
    url: Optional[HttpUrl] = None
    title: str
    content: Optional[str] = None
    file_path: Optional[str] = None
    resource_type: str  # 'url', 'pdf', 'text'
    tags: List[str] = []
    metadata: Dict = {}


class ResourceManager:
    def __init__(self, recreate_index: bool = False):
        """
        Initialize ResourceManager
        Args:
            recreate_index (bool): If True, will drop and recreate the vector search index
        """
        logger.info("Initializing ResourceManager with configuration:")
        logger.info(f"Azure Endpoint: {config.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Embedding Deployment: {config.EMBEDDING_DEPLOYMENT}")

        # Initialize MongoDB connection
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.DB_NAME]
        self.collection = self.db[config.COLLECTION_NAME]

        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            deployment="text-embedding-3-small"
        )

        # Ensure vector search index exists
        self._ensure_vector_search_index(recreate=recreate_index)

        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=config.INDEX_NAME,
            text_key="text"
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

    def _delete_search_index(self):
        """Delete the search index if it exists"""
        try:
            # List all search indexes
            indexes = self.collection.list_search_indexes()
            for index in indexes:
                if index.get('name') == config.INDEX_NAME:
                    logger.info(f"Dropping existing index: {config.INDEX_NAME}")
                    self.collection.drop_search_index(config.INDEX_NAME)
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error deleting search index: {str(e)}")
            return False

    def _ensure_vector_search_index(self, recreate: bool = False):
        """
        Ensure the vector search index exists with correct configuration
        Args:
            recreate (bool): If True, will drop and recreate the index
        """
        try:
            if recreate:
                self._delete_search_index()

            # Define the search index
            search_index = {
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "dimensions": 1536,
                                "similarity": "cosine",
                                "type": "knnVector"
                            },
                            "text": {
                                "type": "string"
                            },
                            "metadata": {
                                "type": "document"
                            }
                        }
                    }
                },
                "name": config.INDEX_NAME
            }

            try:
                # Try to create the index
                self.collection.create_search_index(model=search_index)
                logger.info(f"Created vector search index: {config.INDEX_NAME}")
            except Exception as e:
                if "Duplicate Index" in str(e) or "IndexAlreadyExists" in str(e):
                    logger.info(f"Vector search index {config.INDEX_NAME} already exists")
                else:
                    raise

        except Exception as e:
            logger.error(f"Error ensuring vector search index: {str(e)}")
            raise

    def generate_resource_id(self, resource: Resource) -> str:
        """Generate a unique ID for a resource"""
        content = resource.url or resource.file_path or resource.content
        return hashlib.md5(f"{resource.title}{content}".encode()).hexdigest()

    async def add_resource(self, resource: Resource) -> Dict:
        """Add a new resource to the vector store"""
        try:
            # Check if resource already exists
            resource_id = self.generate_resource_id(resource)
            if self.collection.find_one({"metadata.resource_id": resource_id}):
                return {"status": "error", "message": "Resource already exists"}

            # Process resource based on type
            if resource.resource_type == "url":
                documents = await self._process_url_resource(resource)
            elif resource.resource_type == "pdf":
                documents = await self._process_pdf_resource(resource)
            elif resource.resource_type == "text":
                documents = await self._process_text_resource(resource)
            else:
                return {"status": "error", "message": "Unsupported resource type"}

            # Add metadata to documents
            for doc in documents:
                # Remove any language field from metadata
                if 'language' in doc.metadata:
                    del doc.metadata['language']

                doc.metadata.update({
                    "resource_id": resource_id,
                    "title": resource.title,
                    "type": resource.resource_type,
                    "tags": resource.tags,
                    "added_at": datetime.utcnow().isoformat(),
                    "added_by": "user",
                    **{k: v for k, v in resource.metadata.items() if k != 'language'}
                })

            # Add to vector store using direct MongoDB insertion
            try:
                docs_to_insert = []
                for i, doc in enumerate(documents):
                    embedding = self.embeddings.embed_query(doc.page_content)
                    docs_to_insert.append({
                        "text": doc.page_content,
                        "embedding": embedding,
                        "metadata": doc.metadata
                    })

                if docs_to_insert:
                    self.collection.insert_many(docs_to_insert)

                return {
                    "status": "success",
                    "message": f"Added {len(documents)} chunks to vector store",
                    "resource_id": resource_id
                }
            except Exception as e:
                logger.error(f"Error adding to vector store: {str(e)}")
                return {"status": "error", "message": str(e)}

        except Exception as e:
            logger.error(f"Error adding resource: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _process_url_resource(self, resource: Resource) -> List[Document]:
        """Process URL resource with text cleaning"""
        try:
            # Validate URL
            url = str(resource.url)
            logger.info(f"Processing URL: {url}")

            # Configure WebBaseLoader with headers and timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            loader = WebBaseLoader(
                web_path=url,
                header_template=headers,
                verify_ssl=False,  # Only if needed for specific sites
                requests_per_second=2
            )

            # Load documents with error handling
            try:
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} documents from URL")
            except Exception as load_error:
                logger.error(f"Error loading URL content: {str(load_error)}")
                raise

            if not documents:
                logger.warning("No content loaded from URL")
                return []

            # Clean documents using TextCleaner
            cleaned_docs = []
            for doc in documents:
                try:
                    # Add basic content validation
                    if not doc.page_content or len(doc.page_content.strip()) < 10:
                        logger.warning("Skipping document with insufficient content")
                        continue

                    # Clean the text
                    cleaned_text = TextCleaner.process_text(doc.page_content)

                    # Validate cleaned text
                    if cleaned_text and len(cleaned_text.strip()) >= 10:
                        cleaned_docs.append(Document(
                            page_content=cleaned_text,
                            metadata={
                                **doc.metadata,
                                'source': url,
                                'title': resource.title,
                                'processed_at': datetime.utcnow().isoformat()
                            }
                        ))
                    else:
                        logger.warning("Skipping document after cleaning due to insufficient content")

                except Exception as clean_error:
                    logger.error(f"Error cleaning document: {str(clean_error)}")
                    continue

            if not cleaned_docs:
                logger.warning("No valid documents after cleaning")
                return []

            logger.info(f"Successfully cleaned {len(cleaned_docs)} documents")

            # Clean metadata and split documents
            for doc in cleaned_docs:
                if 'language' in doc.metadata:
                    del doc.metadata['language']

            split_docs = self.text_splitter.split_documents(cleaned_docs)
            logger.info(f"Split into {len(split_docs)} chunks")

            return split_docs

        except Exception as e:
            logger.error(f"Error processing URL resource: {str(e)}", exc_info=True)
            raise

    async def _process_pdf_resource(self, resource: Resource) -> List[Document]:
        """Process PDF resource with text cleaning"""
        try:
            if not resource.file_path:
                raise ValueError("PDF file path is required")

            logger.info(f"Processing PDF file: {resource.file_path}")

            # Using custom PDF loader
            loader = CustomPDFLoader(resource.file_path)
            documents = loader.load()

            logger.info(f"Successfully loaded {len(documents)} pages from PDF")

            # Clean documents using TextCleaner
            cleaned_docs = []
            for doc in documents:
                cleaned_text = TextCleaner.process_text(doc.page_content)
                if cleaned_text and len(cleaned_text.strip()) >= 10:
                    cleaned_docs.append(Document(
                        page_content=cleaned_text,
                        metadata={
                            **doc.metadata,
                            'title': resource.title,
                            'processed_at': datetime.utcnow().isoformat(),
                            'file_path': resource.file_path
                        }
                    ))

            logger.info(f"Successfully cleaned {len(cleaned_docs)} pages")

            # Clean metadata
            for doc in cleaned_docs:
                if 'language' in doc.metadata:
                    del doc.metadata['language']

            split_docs = self.text_splitter.split_documents(cleaned_docs)
            logger.info(f"Split into {len(split_docs)} chunks")

            return split_docs

        except Exception as e:
            logger.error(f"Error processing PDF resource: {str(e)}", exc_info=True)
            raise

    async def _process_text_resource(self, resource: Resource) -> List[Document]:
        """Process text resource with text cleaning"""
        try:
            if resource.content:
                # Clean the content using TextCleaner
                cleaned_text = TextCleaner.process_text(resource.content)
                documents = [Document(page_content=cleaned_text, metadata={})]
            elif resource.file_path:
                loader = TextLoader(resource.file_path)
                documents = loader.load()
                # Clean documents using TextCleaner
                cleaned_docs = []
                for doc in documents:
                    cleaned_text = TextCleaner.process_text(doc.page_content)
                    cleaned_docs.append(Document(
                        page_content=cleaned_text,
                        metadata=doc.metadata
                    ))
                documents = cleaned_docs

                # Clean metadata
                for doc in documents:
                    if 'language' in doc.metadata:
                        del doc.metadata['language']
            else:
                raise ValueError("Either content or file_path must be provided")

            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error processing text resource: {str(e)}")
            raise

    async def delete_resource(self, resource_id: str) -> Dict:
        """Delete a resource from the vector store"""
        try:
            result = self.collection.delete_many({"metadata.resource_id": resource_id})
            return {
                "status": "success",
                "message": f"Deleted {result.deleted_count} documents"
            }
        except Exception as e:
            logger.error(f"Error deleting resource: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def list_resources(self) -> List[Dict]:
        """List all resources in the vector store"""
        try:
            resources = self.collection.aggregate([
                {"$group": {
                    "_id": "$metadata.resource_id",
                    "title": {"$first": "$metadata.title"},
                    "type": {"$first": "$metadata.type"},
                    "tags": {"$first": "$metadata.tags"},
                    "added_at": {"$first": "$metadata.added_at"},
                    "chunk_count": {"$sum": 1}
                }}
            ])
            return list(resources)
        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            return []

    async def search_resources(self, query: str, limit: int = 5) -> List[Dict]:
        """Search resources in the vector store"""
        try:
            results = self.vector_store.similarity_search(query, k=limit)
            return [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in results]
        except Exception as e:
            logger.error(f"Error searching resources: {str(e)}")
            return []

    def test_embedding_connection(self):
        """Test the embedding connection"""
        try:
            test_text = "This is a test sentence."
            logger.info("Attempting to generate embeddings...")
            logger.info(f"Using deployment: text-embedding-3-small")

            embedding = self.embeddings.embed_query(test_text)

            logger.info(f"Successfully generated embedding of length: {len(embedding)}")
            return {
                "status": "success",
                "message": "Successfully generated embeddings",
                "embedding_length": len(embedding),
                "deployment": "text-embedding-3-small"
            }
        except Exception as e:
            logger.error(f"Embedding test failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate embeddings: {str(e)}",
                "deployment": "text-embedding-3-small",
                "endpoint": config.AZURE_OPENAI_ENDPOINT
            }

    def __del__(self):
        """Cleanup temporary files on deletion"""
        try:
            if os.path.exists("temp"):
                for file in os.listdir("temp"):
                    file_path = os.path.join("temp", file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")