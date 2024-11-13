# Alzheimer Expert Bot 🧠

The Alzheimer Expert Bot is an advanced clinical decision support system that combines AI-powered chat capabilities with comprehensive resource management for Alzheimer's disease information. The system utilizes the Claude AI model along with vector search capabilities to provide accurate, evidence-based responses to clinical queries.

## Features 🌟

### Clinical Assistant
- **Dual Search Modes**:
  - Resource-Based Search: Provides responses with citations from curated medical literature
  - Open Clinical Search: Accesses broader medical knowledge base
- **Smart Context Management**: Optimizes response relevance through intelligent context handling
- **Citation System**: Includes IEEE-style citations for all resource-based responses
- **Suggested Questions**: Generates contextually relevant follow-up questions
- **Conversation History**: Maintains and displays full consultation history

### Resource Management
- **Multiple Resource Types Support**:
  - URL-based resources
  - PDF documents
  - Text content
- **Advanced Search Capabilities**:
  - Full-text search across all resources
  - Filtering by resource type
  - Tag-based organization
- **Vector Search Integration**: Uses MongoDB Atlas Vector Search for semantic similarity matching
- **Resource Metadata**: Tracks creation dates, types, tags, and sources

## Technical Architecture 🔧

### Backend (FastAPI)
- **API Framework**: FastAPI with async support
- **Database**: MongoDB with Vector Search capabilities
- **Embedding**: Azure OpenAI embeddings for semantic search
- **AI Model**: Claude AI for natural language processing
- **Text Processing**: Advanced text cleaning and chunking for optimal indexing

### Frontend (Streamlit)
- **UI Framework**: Streamlit with custom styling
- **Responsive Design**: Adaptive layout for different screen sizes
- **Real-time Updates**: Dynamic content loading and state management
- **Interactive Components**: Custom buttons, expandable sections, and tabbed interfaces

## System Components 🔍

### Core Services
1. **Vector Store**: MongoDB Atlas Vector Search for semantic similarity matching
2. **Embeddings Service**: Azure OpenAI for text embeddings
3. **AI Model**: Claude for natural language understanding and generation
4. **Resource Manager**: Handles document processing and storage

### Key Features
1. **Context Optimization**:
   - Token counting and management
   - Intelligent chunk selection
   - Conversation history tracking

2. **Resource Processing**:
   - Automatic text cleaning
   - Metadata extraction
   - Vector embedding generation

3. **Response Generation**:
   - Citation integration
   - Source validation
   - Context-aware responses

## API Endpoints 🛣️

### Chat Endpoints
- `POST /chat`: Process chat messages and generate responses
- `GET /health`: System health check

### Resource Management
- `POST /resources/add`: Add new resources
- `POST /resources/upload-pdf`: Upload PDF documents
- `GET /resources/search`: Search existing resources
- `GET /resources`: List all resources
- `DELETE /resources/{resource_id}`: Remove resources

## Environment Setup 🔧

Required environment variables:
```
MONGODB_URI=your_mongodb_uri
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
ANTHROPIC_API_KEY=your_anthropic_key
```

## Error Handling 🚨

The system includes comprehensive error handling:
- Input validation
- API error responses
- Resource processing errors
- Connection handling
- Token limit management

## Logging 📝

Implements multi-level logging:
- Standard output logging
- Error file logging
- API request logging
- System health monitoring

## Security Measures 🔐

- Authentication for API endpoints
- Secure file handling
- Environment variable protection
- Data validation
- CORS configuration

## Getting Started 🚀

1. Clone the repository
2. Set up environment variables
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
5. Start the frontend:
   ```bash
   streamlit run streamlit_app.py
   ```

## Contributing 🤝

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License 📄

This project is licensed under the MIT License - see the LICENSE.md file for details.

