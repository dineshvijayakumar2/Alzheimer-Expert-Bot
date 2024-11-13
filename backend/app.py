# app.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
import uvicorn
from config import config
from anthropic import Anthropic
import re
import logging
import os

# Import necessary LangChain components
from langchain.vectorstores import MongoDBAtlasVectorSearch
from embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, TextLoader
from resource_manager import ResourceManager, Resource
from pymongo import MongoClient
from bson import ObjectId

from text_cleaning import TextCleaner, preprocess_documents
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add error file handler
error_handler = logging.FileHandler('error.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(error_handler)

# Initialize FastAPI app
app = FastAPI(title="Alzheimer Expert Bot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchMode(str, Enum):
    RESOURCE = "resource"
    GENERAL = "general"

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    search_mode: SearchMode = SearchMode.RESOURCE

class Source(BaseModel):
    title: str = "Source Document"
    url: Optional[str] = None
    citation_index: int
    ieee_citation: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Source] = []
    suggested_questions: List[str] = []

class ResourceBase(BaseModel):
    title: str
    content: Optional[str] = None
    url: Optional[HttpUrl] = None
    type: str = "Clinical"
    tags: List[str] = []

class ResourceResponse(BaseModel):
    id: str
    content: str
    title: str
    type: str
    date_added: datetime
    url: Optional[str] = None
    tags: List[str] = []

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ResourceSearch(BaseModel):
    query: Optional[str] = None
    types: List[str] = ["Clinical", "Research", "Guidelines"]
    sort: str = "Date Added"

# Global variables
vector_store = None
embeddings = None
anthropic_client = None
mongo_client = None
text_splitter = None
resource_manager = None



@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global vector_store, embeddings, anthropic_client, mongo_client, text_splitter, resource_manager

    try:
        # Initialize MongoDB connection
        mongo_client = MongoClient(config.MONGODB_URI)

        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            openai_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-15-preview",
            deployment_name="text-embedding-3-small"
        )

        # Initialize vector store
        collection = mongo_client[config.DB_NAME][config.COLLECTION_NAME]
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=config.INDEX_NAME
        )

        # Initialize Anthropic client
        anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize ResourceManager
        try:
            resource_manager = ResourceManager(recreate_index=False)
            logger.info("ResourceManager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResourceManager: {str(e)}")
            raise

        # Create temp directory for file uploads
        os.makedirs("temp", exist_ok=True)

        # Load initial documents if vector store is empty
        if collection.count_documents({}) == 0:
            await load_initial_documents()

        logger.info("Successfully initialized all components")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise



async def load_initial_documents():
    """Load initial documents into vector store"""
    try:
        base_urls = [
            "https://www.alz.org/alzheimers-dementia/what-is-alzheimers",
            "https://www.alz.org/alzheimers-dementia/treatments",
            "https://www.alz.org/alzheimers-dementia/diagnosis"
        ]

        loader = WebBaseLoader(base_urls)
        raw_documents = loader.load()

        # Process and clean documents
        cleaned_docs = []
        for doc in raw_documents:
            cleaned_text = TextCleaner.process_text(doc.page_content)
            doc.metadata['url'] = doc.metadata.get('source')
            cleaned_docs.append(Document(
                page_content=cleaned_text,
                metadata=doc.metadata
            ))

        # Split documents
        splits = text_splitter.split_documents(cleaned_docs)

        # Add source URLs to metadata
        for i, split in enumerate(splits):
            split.metadata['url'] = base_urls[i // (len(splits) // len(base_urls))]
            split.metadata['source'] = base_urls[i // (len(splits) // len(base_urls))]
            split.metadata['date_added'] = datetime.utcnow()
            split.metadata['type'] = 'Clinical'

        # Add to vector store
        vector_store.add_documents(splits)
        logger.info(f"Loaded {len(splits)} document chunks into vector store")

    except Exception as e:
        logger.error(f"Error loading initial documents: {str(e)}")
        raise


# Chatbot Endpoints and Functions
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with context-aware responses"""
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")

        # Get and optimize context based on search mode
        context, sources = "", []
        if request.search_mode == SearchMode.RESOURCE:
            context, raw_sources = get_relevant_context(request.message)
            context, sources = optimize_context(context, raw_sources)

            if not context.strip():
                return ChatResponse(
                    response="I don't have enough relevant information in my knowledge base. Please try rephrasing your question.",
                    sources=[],
                    suggested_questions=[
                        "What are the early signs of Alzheimer's disease?",
                        "How is Alzheimer's diagnosed?",
                        "What are the current treatment options?"
                    ]
                )

        # Process conversation history and create prompt
        conv_history = trim_conversation_history(request.history)
        prompt = create_optimized_prompt(
            query=request.message,
            context=context,
            history=conv_history,
            search_mode=request.search_mode
        )

        # Get response from Anthropic
        response = anthropic_client.messages.create(
            model=config.MODEL_NAME,
            max_tokens=config.MAX_TOKENS,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        # Format response and generate suggestions
        formatted_response = format_response_with_links(response.content[0].text, sources) \
            if request.search_mode == SearchMode.RESOURCE else response.content[0].text

        suggested_questions = await generate_suggested_questions(request.message, context or formatted_response)
        formatted_sources = format_ieee_citations(sources) if request.search_mode == SearchMode.RESOURCE else []

        return ChatResponse(
            response=formatted_response,
            sources=formatted_sources,
            suggested_questions=suggested_questions
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Resource Management Endpoints
@app.post("/resources/add", response_model=Dict)
async def add_resource(resource: ResourceBase):
    """Add a single resource to the vector store"""
    try:
        # Convert ResourceBase to Resource
        resource_obj = Resource(
            title=resource.title,
            content=resource.content,
            url=resource.url,
            resource_type=resource.type,
            tags=resource.tags,
            metadata={
                "date_added": datetime.utcnow().isoformat(),
                "type": resource.type
            }
        )

        # Add resource using ResourceManager
        result = await resource_manager.add_resource(resource_obj)

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return result

    except Exception as e:
        logger.error(f"Error adding resource: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resources/upload-pdf", response_model=Dict)
async def upload_pdf_resource(
        file: UploadFile = File(...),
        title: str = Form(...),
        type: str = Form(...),
        tags: str = Form("")
):
    """Upload PDF resource"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

        # Generate a secure filename
        file_path = f"temp/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"

        try:
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Create Resource object
            resource = Resource(
                title=title,
                file_path=file_path,
                resource_type="pdf",
                tags=tags.split(",") if tags else [],
                metadata={
                    "added_by": "user",
                    "date_added": datetime.utcnow().isoformat(),
                    "original_filename": file.filename,
                    "file_size": len(content),
                    "type": type
                }
            )

            # Add resource using ResourceManager
            result = await resource_manager.add_resource(resource)

            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])

            return result

        finally:
            # Ensure cleanup happens even if an error occurs
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/search", response_model=List[ResourceResponse])
async def search_resources(
        query: Optional[str] = None,
        types: List[str] = ["Clinical", "Research", "Guidelines"]
):
    """Search resources"""
    try:
        if not query:
            # If no query, list all resources
            resources = await resource_manager.list_resources()
            return [r for r in resources if r["type"] in types]

        # Search with query
        results = await resource_manager.search_resources(query, limit=10)

        # Filter by type if specified
        filtered_results = [
            r for r in results
            if r["metadata"].get("type") in types
        ]

        return filtered_results
    except Exception as e:
        logger.error(f"Error searching resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/resources/{resource_id}")
async def delete_resource(resource_id: str):
    """Delete a resource"""
    try:
        result = await resource_manager.delete_resource(resource_id)

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return result

    except Exception as e:
        logger.error(f"Error deleting resource: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources", response_model=List[Dict])
async def list_resources():
    """List all resources"""
    try:
        resources = await resource_manager.list_resources()
        return resources
    except Exception as e:
        logger.error(f"Error listing resources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/health")
async def health_check():
    """Check system health and component status"""
    try:
        # Test MongoDB connection
        mongo_status = False
        if mongo_client:
            mongo_client.admin.command('ping')
            mongo_status = True

        # Check ResourceManager connection
        resource_manager_status = resource_manager.test_embedding_connection()

        # Check component status
        components_healthy = all([
            mongo_status,
            vector_store is not None,
            embeddings is not None,
            anthropic_client is not None,
            resource_manager is not None
        ])

        return {
            "status": "healthy" if components_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "mongodb": mongo_status,
                "vector_store": vector_store is not None,
                "embeddings": embeddings is not None,
                "anthropic": anthropic_client is not None,
                "resource_manager": resource_manager_status
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

def get_relevant_context(query: str, k: int = 5) -> tuple[str, List[Dict]]:
    """Retrieve relevant context from vector store"""
    try:
        logger.info(f"Retrieving context for query: {query}")
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            logger.warning(f"No relevant documents found for query: {query}")
            return "", []

        context = "\n\n".join([f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)])
        sources = [{
            "title": doc.metadata.get("title", "Unknown"),
            "source": doc.metadata.get("source", ""),  # Original source field
            "url": doc.metadata.get("url", doc.metadata.get("source", "")),  # Try url first, fallback to source
            "content": doc.page_content[:200] + "...",
            "citation_index": i + 1
        } for i, doc in enumerate(docs)]

        logger.info(f"Retrieved {len(docs)} relevant documents")
        return context, sources
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}", exc_info=True)
        return "", []


async def generate_suggested_questions(message: str, context: str) -> List[str]:
    """Generate suggested follow-up questions based on the context and previous interaction"""
    try:
        logger.info("Generating suggested questions")
        prompt = f"""Based on this context about Alzheimer's disease:

Context: {context}

Previous question: {message}

Generate 3 relevant follow-up questions that would help the user learn more about specific aspects mentioned in the context. 
Make questions specific to the provided context.
Each question should focus on different aspects.
Questions should be clear and concise.

Return exactly 3 questions, one per line, without numbering or additional text."""

        result = anthropic_client.messages.create(
            model=config.MODEL_NAME,
            max_tokens=200,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        questions = [q.strip() for q in result.content[0].text.strip().split('\n') if q.strip()]

        while len(questions) < 3:
            questions.append("What other aspects of Alzheimer's disease would you like to learn about?")

        logger.info(f"Generated {len(questions)} suggested questions")
        return questions[:3]

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        return [
            "What other aspects of Alzheimer's disease would you like to learn about?",
            "Would you like to know more about treatment options?",
            "Do you have questions about early signs and symptoms?"
        ]


def format_ieee_citations(sources: List[Dict]) -> List[Dict]:
    """Format sources with simplified IEEE citations, always showing titles"""
    formatted_sources = []
    for i, source in enumerate(sources):
        # Get title (with a meaningful default if not provided)
        title = source.get('title', 'Source Document')
        if title.lower() == 'unknown':
            title = 'Source Document'

        # Get URL (prioritize url field, fallback to source field)
        citation_url = source.get('url') or source.get('source')

        # Build IEEE citation
        ieee_citation = f"[{i + 1}] \"{title}\""

        # Only add URL if available
        if citation_url:
            ieee_citation += f". Available: {citation_url}"

        ieee_citation += f". [Accessed: {datetime.now().strftime('%d-%b-%Y')}]."

        formatted_source = {
            "title": title,
            "url": citation_url,  # URL might be None
            "citation_index": i + 1,
            "ieee_citation": ieee_citation
        }
        formatted_sources.append(formatted_source)

    return formatted_sources


def format_response_with_links(response: str, sources: List[Dict]) -> str:
    """Format response with hyperlinked citations"""
    for source in sources:
        citation = f"[{source['citation_index']}]"
        # Use URL if available, fallback to source for the hyperlink
        link_url = source.get('url') or source.get('source')
        if link_url:
            linked_citation = f"[[{source['citation_index']}]]({link_url})"
            response = response.replace(citation, linked_citation)
    return response


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # encoding for Claude
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}")
        # Fallback to approximate count
        return len(text.split())


def trim_conversation_history(history: List[Dict[str, str]], max_tokens: int = 1000) -> str:
    """Trim conversation history to stay within token limit"""
    if not history:
        return ""

    formatted_messages = []
    current_tokens = 0

    # Process messages from most recent to oldest
    for message in reversed(history[-4:]):  # Only consider last 4 messages
        formatted_msg = f"{'Human' if message['role'] == 'user' else 'Assistant'}: {message['content']}\n"
        msg_tokens = count_tokens(formatted_msg)

        if current_tokens + msg_tokens > max_tokens:
            break

        formatted_messages.insert(0, formatted_msg)
        current_tokens += msg_tokens

    return "\n".join(formatted_messages)


def optimize_context(context: str, sources: List[Dict], max_tokens: int = 2000) -> Tuple[str, List[Dict]]:
    """Optimize context by selecting most relevant chunks within token limit"""
    chunks = context.split("\n\n")
    optimized_chunks = []
    optimized_sources = []
    current_tokens = 0

    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)

        if current_tokens + chunk_tokens > max_tokens:
            break

        optimized_chunks.append(chunk)
        if i < len(sources):  # Make sure we have a corresponding source
            optimized_sources.append(sources[i])
        current_tokens += chunk_tokens

    return "\n\n".join(optimized_chunks), optimized_sources


def create_optimized_prompt(query: str, context: str, history: str, search_mode: str = "resource") -> str:
    """Create an optimized prompt with essential instructions based on search mode"""
    if search_mode == "resource":
        return f"""You are an expert research assistant specializing in Alzheimer's disease studies. Answer using ONLY the provided context.

Previous conversation:
{history}

Context (with citations):
{context}

Current question: {query}

Key requirements:
1. Use ONLY information from the context
2. Include [n] citations for each claim
3. If information is missing, acknowledge it
4. Be concise but accurate
5. Maintain consistency with previous conversation"""
    else:
        return f"""You are an expert research assistant specializing in Alzheimer's disease studies. Provide your expert knowledge to answer the question.

Previous conversation:
{history}

Current question: {query}

Key requirements:
1. Be accurate and evidence-based
2. Be concise but thorough
3. Maintain consistency with previous conversation"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)