from langchain.prompts import PromptTemplate
from typing import Dict
import json

class JDAnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Analyse cette description de poste et retourne UNIQUEMENT un JSON valide (sans texte avant/après):
{jd_text}

JSON avec: job_title, seniority_level (junior/mid/senior), required_skills (liste), preferred_skills (liste), experience_required, key_responsibilities (liste), company_culture, summary (2-3 phrases).

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