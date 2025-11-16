from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
import operator
import concurrent.futures

# Définition de l'état partagé
class InterviewPrepState(TypedDict):
    cv_text: str
    cv_analysis: Dict
    jd_text: str
    jd_analysis: Dict
    company_name: str
    company_info: Dict
    questions: List[Dict]
    current_question_idx: int
    user_answers: List[Dict]
    feedback_history: List[Dict]
    general_tips: Dict
    human_approval_needed: bool
    human_feedback: str
    next_step: str
    error: str

class InterviewPrepSupervisor:
    def __init__(self, agents: Dict, vector_store, memory_saver):
        self.agents = agents
        self.vector_store = vector_store
        self.memory = memory_saver
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Construit le graphe LangGraph"""
        workflow = StateGraph(InterviewPrepState)
        
        # Ajouter les nœuds
        workflow.add_node("analyze_cv", self.analyze_cv_node)
        workflow.add_node("analyze_jd", self.analyze_jd_node)
        workflow.add_node("research_company", self.research_company_node)
        workflow.add_node("generate_questions", self.generate_questions_node)
        workflow.add_node("human_review", self.human_review_node)
        workflow.add_node("conduct_interview", self.conduct_interview_node)
        workflow.add_node("provide_feedback", self.provide_feedback_node)
        workflow.add_node("generate_tips", self.generate_tips_node)
        
        # Définir le flux
        # Exécuter analyze_cv et analyze_jd en parallèle pour gagner du temps
        workflow.set_entry_point("analyze_parallel")
        workflow.add_node("analyze_parallel", self.analyze_parallel_node)
        
        workflow.add_edge("analyze_parallel", "research_company")
        workflow.add_edge("research_company", "generate_questions")
        workflow.add_edge("generate_questions", "human_review")
        
        # Conditional edge après human review
        workflow.add_conditional_edges(
            "human_review",
            self.route_after_human_review,
            {
                "approved": "generate_tips",
                "regenerate": "generate_questions",
                "interview": "conduct_interview"
            }
        )
        
        workflow.add_edge("generate_tips", "conduct_interview")
        workflow.add_edge("conduct_interview", "provide_feedback")
        
        # Conditional edge pour continuer ou terminer
        workflow.add_conditional_edges(
            "provide_feedback",
            self.route_after_feedback,
            {
                "continue": "conduct_interview",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory)
    
    def get_graph(self):
        """Retourne le graph compilé"""
        return self.graph
    
    def analyze_parallel_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud qui exécute l'analyse CV et JD en parallèle pour gagner du temps"""
        def analyze_cv():
            try:
                if not state.get("cv_text") or not state["cv_text"].strip():
                    return {}, "CV text is empty"
                result = self.agents["cv_analyzer"].analyze(state["cv_text"])
                if result.get("success") and result.get("analysis"):
                    return result.get("analysis", {}), ""
                return {}, result.get("error", "CV Analysis failed")
            except Exception as e:
                return {}, f"CV Analysis exception: {str(e)}"
        
        def analyze_jd():
            try:
                if not state.get("jd_text") or not state["jd_text"].strip():
                    return {}, "JD text is empty"
                result = self.agents["jd_analyzer"].analyze(state["jd_text"])
                if result.get("success") and result.get("analysis"):
                    return result.get("analysis", {}), ""
                return {}, result.get("error", "JD Analysis failed")
            except Exception as e:
                return {}, f"JD Analysis exception: {str(e)}"
        
        # Exécuter les deux analyses en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            cv_future = executor.submit(analyze_cv)
            jd_future = executor.submit(analyze_jd)
            
            cv_analysis, cv_error = cv_future.result()
            jd_analysis, jd_error = jd_future.result()
        
        # Stocker dans vector DB si succès
        if cv_analysis:
            try:
                self.vector_store.add_documents(
                    "cv_data",
                    [state["cv_text"]],
                    [{"type": "cv", "analysis": cv_analysis}]
                )
            except Exception:
                pass
        
        if jd_analysis:
            try:
                self.vector_store.add_documents(
                    "jd_data",
                    [state["jd_text"]],
                    [{"type": "jd", "analysis": jd_analysis}]
                )
            except Exception:
                pass
        
        # Combiner les erreurs
        errors = []
        if cv_error:
            errors.append(f"CV: {cv_error}")
        if jd_error:
            errors.append(f"JD: {jd_error}")
        
        return {
            **state,
            "cv_analysis": cv_analysis,
            "jd_analysis": jd_analysis,
            "error": "; ".join(errors) if errors else ""
        }
    
    def analyze_cv_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'analyse CV"""
        try:
            # Vérifier que cv_text n'est pas vide
            if not state.get("cv_text") or not state["cv_text"].strip():
                return {
                    **state,
                    "cv_analysis": {},
                    "error": "CV text is empty"
                }
            
            result = self.agents["cv_analyzer"].analyze(state["cv_text"])
            
            # Debug: vérifier la structure de result
            if not isinstance(result, dict):
                return {
                    **state,
                    "cv_analysis": {},
                    "error": f"CV Analysis returned invalid result: {type(result)}"
                }
            
            # Vérifier que result contient bien les données
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                return {
                    **state,
                    "cv_analysis": {},
                    "error": f"CV Analysis failed: {error_msg}"
                }
            
            analysis = result.get("analysis", {})
            
            # Vérifier que analysis n'est pas vide
            if not analysis or (isinstance(analysis, dict) and len(analysis) == 0):
                return {
                    **state,
                    "cv_analysis": {},
                    "error": "CV Analysis returned empty result"
                }
            
            if result["success"] and analysis:
                # Stocker dans vector DB
                try:
                    self.vector_store.add_documents(
                        "cv_data",
                        [state["cv_text"]],
                        [{"type": "cv", "analysis": analysis}]
                    )
                except Exception as e:
                    # Ne pas bloquer si le vector store échoue
                    pass
            
            return {
                **state,
                "cv_analysis": analysis,
                "error": ""
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return {
                **state,
                "cv_analysis": {},
                "error": f"CV Analysis exception: {str(e)}\n{error_trace}"
            }
    
    def analyze_jd_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'analyse JD"""
        try:
            result = self.agents["jd_analyzer"].analyze(state["jd_text"])
            
            # Debug: vérifier la structure de result
            if not isinstance(result, dict):
                return {
                    **state,
                    "jd_analysis": {},
                    "error": f"JD Analysis returned invalid result: {type(result)}"
                }
            
            # Vérifier que result contient bien les données
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                return {
                    **state,
                    "jd_analysis": {},
                    "error": f"JD Analysis failed: {error_msg}"
                }
            
            analysis = result.get("analysis", {})
            
            # Vérifier que analysis n'est pas vide
            if not analysis or (isinstance(analysis, dict) and len(analysis) == 0):
                return {
                    **state,
                    "jd_analysis": {},
                    "error": "JD Analysis returned empty result"
                }
            
            if result["success"] and analysis:
                # Stocker dans vector DB
                try:
                    self.vector_store.add_documents(
                        "jd_data",
                        [state["jd_text"]],
                        [{"type": "jd", "analysis": analysis}]
                    )
                except Exception as e:
                    # Ne pas bloquer si le vector store échoue
                    pass
            
            return {
                **state,
                "jd_analysis": analysis,
                "error": ""
            }
        except Exception as e:
            return {
                **state,
                "jd_analysis": {},
                "error": f"JD Analysis exception: {str(e)}"
            }
    
    def research_company_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de recherche entreprise"""
        industry = state["jd_analysis"].get("industry", "")
        result = self.agents["company_researcher"].research(
            state["company_name"],
            industry
        )
        
        return {
            **state,
            "company_info": result.get("info", {}),
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def generate_questions_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de génération de questions"""
        result = self.agents["question_generator"].generate_questions(
            state["cv_analysis"],
            state["jd_analysis"],
            state["company_info"]
        )
        
        return {
            **state,
            "questions": result.get("questions", []),
            "current_question_idx": 0,
            "human_approval_needed": True,
            "error": "" if result["success"] else result.get("error", "")
        }
    
    def human_review_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud d'attente de validation humaine"""
        # Ce nœud met en pause le workflow pour intervention humaine
        return {
            **state,
            "next_step": "awaiting_human_input"
        }
    
    def conduct_interview_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de simulation d'entretien"""
        # Retourne l'état avec la question courante
        # L'interaction réelle se fait dans l'UI
        return state
    
    def provide_feedback_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de feedback sur une réponse"""
        if not state["user_answers"]:
            return state
        
        last_answer = state["user_answers"][-1]
        question = state["questions"][last_answer["question_idx"]]
        
        context = {
            "cv": state["cv_analysis"],
            "jd": state["jd_analysis"]
        }
        
        result = self.agents["interview_coach"].evaluate_answer(
            question["question"],
            last_answer["answer"],
            context
        )
        
        feedback_history = state.get("feedback_history", [])
        if result["success"]:
            feedback_history.append({
                "question_idx": last_answer["question_idx"],
                "feedback": result["feedback"]
            })
        
        return {
            **state,
            "feedback_history": feedback_history,
            "current_question_idx": state["current_question_idx"] + 1
        }
    
    def generate_tips_node(self, state: InterviewPrepState) -> InterviewPrepState:
        """Nœud de génération de conseils généraux"""
        result = self.agents["interview_coach"].generate_general_tips(
            state["cv_analysis"],
            state["jd_analysis"]
        )
        
        return {
            **state,
            "general_tips": result.get("tips", {})
        }
    
    def route_after_human_review(self, state: InterviewPrepState) -> str:
        """Détermine la route après validation humaine"""
        feedback = state.get("human_feedback", "approved")
        
        if feedback == "regenerate":
            return "regenerate"
        elif feedback == "interview":
            return "interview"
        else:
            return "approved"
    
    def route_after_feedback(self, state: InterviewPrepState) -> str:
        """Détermine si on continue l'entretien ou on termine"""
        if state["current_question_idx"] >= len(state["questions"]):
            return "end"
        return "continue"