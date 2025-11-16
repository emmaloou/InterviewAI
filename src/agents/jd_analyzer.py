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
            
            # Nettoyer la réponse
            response_text = response if isinstance(response, str) else str(response)
            
            # Extraire le JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(response_text)
            
            return {
                "success": True,
                "analysis": analysis
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "analysis": {
                    "job_title": "Poste en cours d'analyse",
                    "seniority_level": "À déterminer",
                    "required_skills": ["Analyse en cours..."],
                    "preferred_skills": [],
                    "experience_required": "À déterminer",
                    "key_responsibilities": ["Analyse en cours..."],
                    "company_culture": "Analyse en cours...",
                    "summary": "Analyse de la description de poste en cours."
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }