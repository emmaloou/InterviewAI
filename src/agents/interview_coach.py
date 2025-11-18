from typing import Dict, Optional, List
import json
from src.utils.json_utils import safe_json_loads

class InterviewCoachAgent:
    def __init__(self, llm, callbacks: Optional[List] = None, langfuse_monitor=None):
        self.llm = llm
        self.callbacks = callbacks or []
        self.langfuse_monitor = langfuse_monitor

    def _extract_json_payload(self, raw_text: str) -> str:
        """Nettoie la réponse du LLM pour ne garder que le JSON"""
        cleaned = (raw_text or "").strip()
        if not cleaned:
            raise ValueError("Réponse LLM vide - impossibilité de parser.")
        
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned

    def _parse_json(self, payload: str) -> Dict:
        """Parse un JSON en tolérant quelques erreurs courantes des LLM"""
        return safe_json_loads(payload)
        
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
            config = {"callbacks": self.callbacks} if self.callbacks else {}
            
            # Utiliser le callback comme context manager si possible
            if self.callbacks and len(self.callbacks) > 0:
                callback = self.callbacks[0]
                if hasattr(callback, '__enter__') and hasattr(callback, '__exit__'):
                    with callback:
                        response = self.llm.invoke(prompt, config=config)
                else:
                    response = self.llm.invoke(prompt, config=config)
            else:
                response = self.llm.invoke(prompt, config=config)
            
            # Flush les callbacks pour s'assurer que les données sont envoyées à Langfuse
            if self.callbacks:
                for callback in self.callbacks:
                    if hasattr(callback, 'flush'):
                        try:
                            callback.flush()
                        except Exception:
                            pass
            
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            payload = self._extract_json_payload(response_text)
            feedback = self._parse_json(payload)
            
            # Logger manuellement l'output dans Langfuse si disponible
            if self.langfuse_monitor:
                try:
                    self.langfuse_monitor.log_agent_execution(
                        agent_name="interview_coach_evaluate",
                        input_data={"question": question[:200], "answer": answer[:500]},
                        output_data={"feedback": feedback}
                    )
                except Exception:
                    pass
            
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
            config = {"callbacks": self.callbacks} if self.callbacks else {}
            
            # Utiliser le callback comme context manager si possible
            if self.callbacks and len(self.callbacks) > 0:
                callback = self.callbacks[0]
                if hasattr(callback, '__enter__') and hasattr(callback, '__exit__'):
                    with callback:
                        response = self.llm.invoke(prompt, config=config)
                else:
                    response = self.llm.invoke(prompt, config=config)
            else:
                response = self.llm.invoke(prompt, config=config)
            
            # Flush les callbacks pour s'assurer que les données sont envoyées à Langfuse
            if self.callbacks:
                for callback in self.callbacks:
                    if hasattr(callback, 'flush'):
                        try:
                            callback.flush()
                        except Exception:
                            pass
            
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            payload = self._extract_json_payload(response_text)
            tips = self._parse_json(payload)
            
            # Logger manuellement l'output dans Langfuse si disponible
            if self.langfuse_monitor:
                try:
                    self.langfuse_monitor.log_agent_execution(
                        agent_name="interview_coach_tips",
                        input_data={"cv_analysis": str(cv_analysis)[:500], "jd_analysis": str(jd_analysis)[:500]},
                        output_data={"tips": tips}
                    )
                except Exception:
                    pass
            
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