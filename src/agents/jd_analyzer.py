from langchain.prompts import PromptTemplate
from typing import Dict
import json

class JDAnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Tu es un expert RH analysant une description de poste.
            
            Description de poste:
            {jd_text}
            
            Extrais en JSON:
            - job_title: titre du poste
            - seniority_level: niveau (junior/mid/senior)
            - required_skills: compétences requises (liste)
            - preferred_skills: compétences souhaitées (liste)
            - experience_required: expérience demandée
            - key_responsibilities: responsabilités principales (liste)
            - company_culture: indices sur la culture
            - summary: résumé en 2-3 phrases
            
            JSON:"""
        )
    
    def analyze(self, jd_text: str) -> Dict:
        """Analyse une description de poste"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"jd_text": jd_text})
            
            analysis = json.loads(response)
            
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }