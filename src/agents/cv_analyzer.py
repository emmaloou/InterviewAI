from langchain.prompts import PromptTemplate
from typing import Dict, Optional, List
from src.utils.json_utils import safe_json_loads

class CVAnalyzerAgent:
    def __init__(self, llm, callbacks: Optional[List] = None, langfuse_monitor=None):
        self.llm = llm
        self.callbacks = callbacks or []
        self.langfuse_monitor = langfuse_monitor
        self.prompt = PromptTemplate.from_template(
            """Analyse ce CV et retourne UNIQUEMENT un JSON valide (sans texte avant/après):
{cv_text}

JSON avec: skills (liste), experience_years (nombre), experience_domains (liste), education (texte), strengths (3-5 points), areas_for_improvement (2-3 points), summary (2-3 phrases).

JSON:"""
        )
    
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

        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

        return cleaned

    def analyze(self, cv_text: str) -> Dict:
        """Analyse un CV"""
        try:
            chain = self.prompt | self.llm
            config = {"callbacks": self.callbacks} if self.callbacks else {}
            
            # Utiliser le callback comme context manager si possible
            if self.callbacks and len(self.callbacks) > 0:
                callback = self.callbacks[0]
                # Si le callback supporte le context manager, l'utiliser
                if hasattr(callback, '__enter__') and hasattr(callback, '__exit__'):
                    with callback:
                        response = chain.invoke({"cv_text": cv_text}, config=config)
                else:
                    response = chain.invoke({"cv_text": cv_text}, config=config)
            else:
                response = chain.invoke({"cv_text": cv_text}, config=config)
            
            # Flush les callbacks pour s'assurer que les données sont envoyées à Langfuse
            if self.callbacks:
                for callback in self.callbacks:
                    if hasattr(callback, 'flush'):
                        try:
                            callback.flush()
                        except Exception:
                            pass
            
            # Nettoyer la réponse
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            
            # Logger manuellement l'output dans Langfuse si disponible
            if self.langfuse_monitor:
                try:
                    self.langfuse_monitor.log_agent_execution(
                        agent_name="cv_analyzer",
                        input_data={"cv_text": cv_text[:500]},  # Limiter la taille
                        output_data={"response": response_text[:2000]}  # Limiter la taille
                    )
                except Exception:
                    pass  # Ne pas bloquer si le logging échoue
            
            # Extraire et parser le JSON
            payload = self._extract_json_payload(response_text)
            analysis = safe_json_loads(payload)
            
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