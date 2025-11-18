import json
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

class LangfuseMonitoring:
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
    def get_callback_handler(self, trace_name: str, user_id: str = None):
        """Crée un callback handler pour tracer les opérations"""
        handler = CallbackHandler(
            trace_name=trace_name,
            user_id=user_id,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        return handler
    
    def _format_payload(self, payload):
        """Convertit un payload en chaîne JSON pour l'affichage Langfuse"""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload)

    def log_agent_execution(self, agent_name: str, input_data: dict, output_data: dict, metadata: dict = None):
        """Log l'exécution d'un agent"""
        metadata = metadata or {}
        trace = self.langfuse.trace(
            name=f"agent_{agent_name}",
            metadata={
                **metadata,
                "input_raw": input_data,
                "output_raw": output_data,
            }
        )
        
        trace.generation(
            name=agent_name,
            input=self._format_payload(input_data),
            output=self._format_payload(output_data),
            metadata=metadata
        )
        
        return trace
    
    def log_workflow_step(self, step_name: str, state: dict, success: bool):
        """Log une étape du workflow"""
        self.langfuse.trace(
            name=f"workflow_step_{step_name}",
            metadata={
                "step": step_name,
                "success": success,
                "state_keys": list(state.keys())
            }
        )