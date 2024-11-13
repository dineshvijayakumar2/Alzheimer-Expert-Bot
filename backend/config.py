# config.py
from dotenv import load_dotenv
import os
from pydantic import BaseModel

# Load environment variables
load_dotenv()


class EnvConfig(BaseModel):
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv('MONGODB_URI')
    DB_NAME: str = os.getenv('DB_NAME')
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME')
    INDEX_NAME: str = os.getenv('INDEX_NAME')
    
    # AI Service Keys
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY')
    AZURE_OPENAI_API_KEY: str = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT: str = os.getenv('AZURE_OPENAI_ENDPOINT')
    EMBEDDING_DEPLOYMENT: str = 'text-embedding-3-small'
    
    # Application Settings
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'claude-3-haiku-20240307')
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', '1000'))
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.2'))

    API_URL: str = os.getenv('API_URL', 'http://localhost:8000/api')

    def validate_config(self) -> None:
        missing_vars = []
        for field, value in self:
            if value is None:
                missing_vars.append(field)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize config
config = EnvConfig()

# Validate on import
config.validate_config()
