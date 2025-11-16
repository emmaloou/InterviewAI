from typing import Dict
import json

class InterviewCoachAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def evaluate_answer(self, question: str, answer: str, 
                       context: Dict) -> Dict:
        """Évalue une réponse et donne du feedback"""
        
        prompt = f"""Tu es un coach d'entretien bienveillant. Évalue cette réponse:

QUESTION: {question}

RÉPONSE DU CANDIDAT:
{answer}

CONTEXTE (CV/Poste):
{json.dumps(context, indent=2)}

Fournis en JSON:
- score: note sur 10
- positive_points: liste de 2-3 points positifs
- improvement_areas: liste de 2-3 points à améliorer
- improved_answer: suggestion de réponse améliorée
- specific_tips: conseils spécifiques (liste de 2-3)
- encouragement: message d'encouragement personnalisé

Sois constructif, spécifique et encourageant.

JSON:"""

        try:
            response = self.llm.invoke(prompt)
            feedback = json.loads(response)
            
            return {
                "success": True,
                "feedback": feedback
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "feedback": {}
            }
    
    def generate_general_tips(self, cv_analysis: Dict, jd_analysis: Dict) -> Dict:
        """Génère des conseils généraux de préparation"""
        
        prompt = f"""Génère des conseils personnalisés de préparation d'entretien.

PROFIL: {json.dumps(cv_analysis, indent=2)}
POSTE: {json.dumps(jd_analysis, indent=2)}

Fournis en JSON:
- preparation_checklist: liste de 5-7 actions à faire
- strengths_to_highlight: points forts à mettre en avant
- potential_concerns: points qui pourraient inquiéter le recruteur + comment les adresser
- dress_code: conseils vestimentaires
- body_language: conseils de langage corporel
- common_mistakes: erreurs à éviter (liste)

JSON:"""

        try:
            response = self.llm.invoke(prompt)
            tips = json.loads(response)
            
            return {
                "success": True,
                "tips": tips
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tips": {}
            }