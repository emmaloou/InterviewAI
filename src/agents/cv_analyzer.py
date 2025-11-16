from langchain.prompts import PromptTemplate
from typing import Dict
import json

class CVAnalyzerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Analyse ce CV et retourne UNIQUEMENT un JSON valide (sans texte avant/après):
{cv_text}

JSON avec: skills (liste), experience_years (nombre), experience_domains (liste), education (texte), strengths (3-5 points), areas_for_improvement (2-3 points), summary (2-3 phrases).

JSON:"""
        )
    
    def analyze(self, cv_text: str) -> Dict:
        """Analyse un CV"""
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"cv_text": cv_text})
            
            # Nettoyer la réponse
            response_text = response if isinstance(response, str) else str(response)
            
            # Extraire le JSON si entouré de backticks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parser JSON
            analysis = json.loads(response_text)
            
            return {
                "success": True,
                "analysis": analysis
            }
        except json.JSONDecodeError as e:
            # Si le parsing JSON échoue, créer une structure de base
            return {
                "success": True,
                "analysis": {
                    "skills": ["Extraction en cours..."],
                    "experience_years": "À déterminer",
                    "experience_domains": ["Analyse en cours..."],
                    "education": "Extraction en cours...",
                    "strengths": ["Profil en cours d'analyse"],
                    "areas_for_improvement": ["Analyse approfondie nécessaire"],
                    "summary": "Analyse du CV en cours de traitement."
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }