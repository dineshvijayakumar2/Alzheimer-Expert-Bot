import re
from bs4 import BeautifulSoup
import unicodedata
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags and clean up HTML entities"""
        try:
            if not text:
                return ""
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(separator=' ')
        except Exception as e:
            logger.warning(f"HTML cleaning error: {str(e)}")
            return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace and remove extra spaces"""
        try:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove spaces before punctuation
            text = re.sub(r'\s+([.,;!?])', r'\1', text)
            return text.strip()
        except Exception as e:
            logger.warning(f"Whitespace normalization error: {str(e)}")
            return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters"""
        try:
            return unicodedata.normalize('NFKC', text)
        except Exception as e:
            logger.warning(f"Unicode normalization error: {str(e)}")
            return text

    @staticmethod
    def clean_special_chars(text: str) -> str:
        """Remove or replace special characters while preserving meaning"""
        try:
            # Replace fancy quotes with straight quotes
            text = re.sub(r'[""]', '"', text)
            text = re.sub(r'['']', "'", text)
            # Replace other special characters
            text = re.sub(r'[−‐‑‒–—―]', '-', text)
            return text
        except Exception as e:
            logger.warning(f"Special character cleaning error: {str(e)}")
            return text

    @staticmethod
    def fix_sentence_boundaries(text: str) -> str:
        """Fix sentence boundaries and spacing"""
        try:
            # Add space after period if missing
            text = re.sub(r'\.(?=[A-Z])', '. ', text)
            # Fix spacing around citations
            text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
            return text
        except Exception as e:
            logger.warning(f"Sentence boundary fixing error: {str(e)}")
            return text

    @staticmethod
    def remove_references(text: str) -> str:
        """Remove reference markers and clean up citations"""
        try:
            # Remove various forms of reference markers
            text = re.sub(r'\[\d+\]|\(\d+\)|<\d+>', '', text)
            return text
        except Exception as e:
            logger.warning(f"Reference removal error: {str(e)}")
            return text

    @staticmethod
    def clean_medical_text(text: str) -> str:
        """Clean medical text while preserving technical terms"""
        try:
            # Standardize common medical abbreviations
            replacements = {
                r'\bAD\b': "Alzheimer's disease",
                r'\bPt\b': "Patient",
                r'\bSx\b': "Symptoms",
                r'\bDx\b': "Diagnosis",
                r'\bTx\b': "Treatment",
                r'\bMCI\b': "Mild Cognitive Impairment",
                r'\bADL\b': "Activities of Daily Living",
                r'\bBPSD\b': "Behavioral and Psychological Symptoms of Dementia"
            }
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
            return text
        except Exception as e:
            logger.warning(f"Medical text cleaning error: {str(e)}")
            return text

    @classmethod
    def process_text(cls, text: str) -> str:
        """Apply all cleaning steps in sequence with error handling"""
        try:
            if not text or not isinstance(text, str):
                return ""
                
            # Basic text cleaning
            text = text.strip()
            
            # Apply cleaning steps with error handling
            try:
                text = cls.clean_html(text)
            except Exception as e:
                logger.warning(f"HTML cleaning error: {str(e)}")
                
            try:
                text = cls.normalize_unicode(text)
            except Exception as e:
                logger.warning(f"Unicode normalization error: {str(e)}")
                
            try:
                text = cls.clean_special_chars(text)
            except Exception as e:
                logger.warning(f"Special character cleaning error: {str(e)}")
                
            try:
                text = cls.normalize_whitespace(text)
            except Exception as e:
                logger.warning(f"Whitespace normalization error: {str(e)}")
                
            try:
                text = cls.fix_sentence_boundaries(text)
            except Exception as e:
                logger.warning(f"Sentence boundary fixing error: {str(e)}")
                
            try:
                text = cls.remove_references(text)
            except Exception as e:
                logger.warning(f"Reference removal error: {str(e)}")
                
            try:
                text = cls.clean_medical_text(text)
            except Exception as e:
                logger.warning(f"Medical text cleaning error: {str(e)}")
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            return ""

    @classmethod
    def process_document(cls, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document dictionary"""
        try:
            if 'page_content' in document:
                document['page_content'] = cls.process_text(document['page_content'])
            return document
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return document

def preprocess_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess a list of documents"""
    try:
        return [TextCleaner.process_document(doc) for doc in documents]
    except Exception as e:
        logger.error(f"Error preprocessing documents: {str(e)}")
        return documents
