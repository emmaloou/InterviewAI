from typing import Dict, List, Optional
import json
from src.utils.json_utils import safe_json_loads

class CompanyResearcherAgent:
    def __init__(self, llm, web_search_tool, callbacks: Optional[List] = None, langfuse_monitor=None):
        self.llm = llm
        self.web_search = web_search_tool
        self.callbacks = callbacks or []
        self.langfuse_monitor = langfuse_monitor

    def _extract_json_payload(self, raw_text: str) -> str:
        """Nettoie la réponse du LLM pour ne garder que le JSON"""
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

    def _parse_json(self, payload: str) -> Dict:
        """Parse un JSON en tolérant quelques erreurs courantes des LLM"""
        return safe_json_loads(payload)
        
    def research(self, company_name: str, industry: str = "") -> Dict:
        """Recherche des informations sur une entreprise"""
        # 1. Recherche web
        search_results = self.web_search.search_company_info(company_name, industry)
        
        if not search_results["success"]:
            return {
                "success": False,
                "error": "Web search failed",
                "info": {}
            }
        
        # 2. Synthèse avec LLM
        results_text = "\n\n".join([
            f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}"
            for r in search_results["results"][:3]
        ])
        
        prompt = f"""Synthétise les informations suivantes sur l'entreprise {company_name}:

{results_text}

Retourne un JSON avec:
- company_name: nom
- main_activity: activité principale
- recent_news: actualités récentes (liste de 2-3 points)
- company_culture: culture d'entreprise
- values: valeurs (liste)
- industry_challenges: défis du secteur (liste)
- interesting_facts: faits intéressants (liste)

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
            info = self._parse_json(payload)
            
            # Logger manuellement l'output dans Langfuse si disponible
            if self.langfuse_monitor:
                try:
                    self.langfuse_monitor.log_agent_execution(
                        agent_name="company_researcher",
                        input_data={"company_name": company_name, "industry": industry},
                        output_data={"info": info, "sources": [r.get("url") for r in search_results["results"][:3]]}
                    )
                except Exception:
                    pass
            
            return {
                "success": True,
                "info": info,
                "sources": [r.get("url") for r in search_results["results"][:3]]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "info": {}
            }