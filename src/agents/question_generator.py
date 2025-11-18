from typing import Dict, List, Optional
import json
from src.utils.json_utils import safe_json_loads

class QuestionGeneratorAgent:
    def __init__(self, llm, callbacks: Optional[List] = None, langfuse_monitor=None):
        self.llm = llm
        self.callbacks = callbacks or []
        self.langfuse_monitor = langfuse_monitor
        
    def _compact_json(self, data: Dict, max_chars: int = 1500) -> str:
        """Compacte un dictionnaire en JSON limité pour réduire le coût LLM"""
        serialized = json.dumps(data or {}, ensure_ascii=False, separators=(",", ":"))
        return serialized[:max_chars]

    def _extract_json_payload(self, raw_text: str) -> str:
        """Nettoie la réponse du LLM pour extraire uniquement le JSON"""
        cleaned = (raw_text or "").strip()
        if not cleaned:
            raise ValueError("Réponse LLM vide - impossibilité de parser le JSON.")
        
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()

        # Dernier recours : extraire entre la première { et la dernière }
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned

    def _parse_json(self, payload: str) -> Dict:
        """Parse un JSON en tolérant quelques erreurs courantes des LLM"""
        return safe_json_loads(payload)
        
    def generate_questions(self, cv_analysis: Dict, jd_analysis: Dict, 
                          company_info: Dict) -> Dict:
        """Génère des questions d'entretien personnalisées"""
        
        cv_snippet = self._compact_json(cv_analysis)
        jd_snippet = self._compact_json(jd_analysis)
        company_snippet = self._compact_json(company_info, max_chars=800)
        
        prompt = f"""Tu es un coach d'entretien. Génère 6 questions personnalisées au format JSON.

PROFIL:
{cv_snippet}

POSTE:
{jd_snippet}

ENTREPRISE:
{company_snippet}

Catégories (2 questions chacune):
1. technique
2. comportementale
3. entreprise/situation

Pour chaque question, retourne:
- category (technique/comportementale/entreprise)
- question
- objective (intention du recruteur)
- tips (liste de 2 conseils concrets)
- difficulty (easy/medium/hard)

Réponds uniquement avec un JSON valide: {{"questions": [...]}}"""

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
            questions_data = self._parse_json(payload)
            
            # Logger manuellement l'output dans Langfuse si disponible
            if self.langfuse_monitor:
                try:
                    self.langfuse_monitor.log_agent_execution(
                        agent_name="question_generator",
                        input_data={"cv_analysis": cv_snippet[:500], "jd_analysis": jd_snippet[:500]},
                        output_data={"questions": questions_data.get("questions", [])[:6]}
                    )
                except Exception:
                    pass
            
            return {
                "success": True,
                "questions": questions_data.get("questions", [])[:6]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "questions": []
            }