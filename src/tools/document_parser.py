import PyPDF2
import docx
from typing import Dict, Optional

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extrait le texte d'un PDF"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Erreur parsing PDF: {str(e)}")
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Extrait le texte d'un DOCX"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Erreur parsing DOCX: {str(e)}")
    
    @staticmethod
    def parse_document(file_path: str) -> Dict:
        """Parse un document et retourne le contenu"""
        extension = file_path.split('.')[-1].lower()
        
        if extension == 'pdf':
            content = DocumentParser.parse_pdf(file_path)
        elif extension in ['docx', 'doc']:
            content = DocumentParser.parse_docx(file_path)
        elif extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            raise ValueError(f"Format non supporté: {extension}")
        
        return {
            "content": content,
            "file_path": file_path,
            "extension": extension,
            "length": len(content)
        }