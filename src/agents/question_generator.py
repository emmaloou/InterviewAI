from typing import Dict, List
import json

class QuestionGeneratorAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def generate_questions(self, cv_analysis: Dict, jd_analysis: Dict, 
                          company_info: Dict) -> Dict:
        """Génère des questions d'entretien personnalisées"""
        
        prompt = f"""Génère 10 questions d'entretien pertinentes et personnalisées.

PROFIL CANDIDAT:
{json.dumps(cv_analysis, indent=2)}

POSTE VISÉ:
{json.dumps(jd_analysis, indent=2)}

ENTREPRISE:
{json.dumps(company_info, indent=2)}

Crée des questions dans ces catégories:
1. Questions techniques (3) - basées sur compétences requises
2. Questions comportementales (3) - évaluation soft skills
3. Questions sur l'entreprise (2) - montrer intérêt
4. Questions de mise en situation (2) - cas pratiques

Pour chaque question, fournis en JSON:
- category: catégorie
- question: la question
- objective: ce que le recruteur évalue
- tips: conseils pour bien répondre (2-3 points)
- difficulty: easy/medium/hard

Format: JSON avec liste "questions"
JSON:"""

        try:
            response = self.llm.invoke(prompt)
            questions_data = json.loads(response)
            
            return {
                "success": True,
                "questions": questions_data.get("questions", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "questions": []
            }