import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.cv_analyzer import CVAnalyzerAgent
from src.agents.jd_analyzer import JDAnalyzerAgent
from src.agents.company_researcher import CompanyResearcherAgent
from src.agents.question_generator import QuestionGeneratorAgent
from src.agents.interview_coach import InterviewCoachAgent
from src.agents.supervisor import InterviewPrepSupervisor, InterviewPrepState
from src.tools.document_parser import DocumentParser
from src.tools.web_search import WebSearchTool
from src.tools.vector_store import VectorStore
from src.utils.llm_config import LLMConfig
from langgraph.checkpoint.sqlite import SqliteSaver

# Configuration de la page
st.set_page_config(
    page_title="InterviewMaster AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session
def init_session_state():
    """Initialise les variables de session"""
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = "upload"
    if "agents_initialized" not in st.session_state:
        st.session_state.agents_initialized = False
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0

def initialize_agents():
    """Initialise tous les agents et outils"""
    if st.session_state.agents_initialized:
        return
    
    with st.spinner("🚀 Initialisation des agents IA..."):
        try:
            # LLM et embeddings
            llm = LLMConfig.get_llm()
            embeddings = LLMConfig.get_embeddings()
            
            # Outils
            web_search = WebSearchTool()
            vector_store = VectorStore(embeddings)
            
            # Agents
            agents = {
                "cv_analyzer": CVAnalyzerAgent(llm),
                "jd_analyzer": JDAnalyzerAgent(llm),
                "company_researcher": CompanyResearcherAgent(llm, web_search),
                "question_generator": QuestionGeneratorAgent(llm),
                "interview_coach": InterviewCoachAgent(llm)
            }
            
            # Memory saver
            memory = SqliteSaver.from_conn_string(":memory:")
            
            # Supervisor
            supervisor = InterviewPrepSupervisor(agents, vector_store, memory)
            
            # Stocker dans session state
            st.session_state.supervisor = supervisor
            st.session_state.vector_store = vector_store
            st.session_state.agents_initialized = True
            
            st.success("✅ Agents initialisés avec succès!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            st.stop()

def upload_documents_section():
    """Section d'upload des documents"""
    st.markdown('<div class="step-header"><h2>📄 Étape 1: Upload des Documents</h2></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CV du Candidat")
        cv_file = st.file_uploader(
            "Uploadez votre CV",
            type=["pdf", "docx", "txt"],
            key="cv_upload"
        )
        
        if cv_file:
            st.success(f"✅ CV uploadé: {cv_file.name}")
    
    with col2:
        st.markdown("### Description de Poste")
        jd_option = st.radio(
            "Comment fournir la description de poste?",
            ["📝 Saisie manuelle", "📄 Upload fichier"]
        )
        
        if jd_option == "📄 Upload fichier":
            jd_file = st.file_uploader(
                "Uploadez la description de poste",
                type=["pdf", "docx", "txt"],
                key="jd_upload"
            )
            jd_text = None
        else:
            jd_text = st.text_area(
                "Collez la description de poste",
                height=200,
                key="jd_text"
            )
            jd_file = None
    
    st.markdown("### 🏢 Informations sur l'Entreprise")
    company_name = st.text_input("Nom de l'entreprise", key="company_name")
    
    # Bouton de démarrage
    if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
        if not cv_file:
            st.error("⚠️ Veuillez uploader un CV")
            return
        
        if not jd_file and not jd_text:
            st.error("⚠️ Veuillez fournir une description de poste")
            return
        
        if not company_name:
            st.error("⚠️ Veuillez entrer le nom de l'entreprise")
            return
        
        # Parser les documents
        try:
            # Sauvegarder temporairement le CV
            cv_path = f"./data/cv/{cv_file.name}"
            Path("./data/cv").mkdir(parents=True, exist_ok=True)
            with open(cv_path, "wb") as f:
                f.write(cv_file.getbuffer())
            
            cv_content = DocumentParser.parse_document(cv_path)["content"]
            
            # Parser JD
            if jd_file:
                jd_path = f"./data/jd/{jd_file.name}"
                Path("./data/jd").mkdir(parents=True, exist_ok=True)
                with open(jd_path, "wb") as f:
                    f.write(jd_file.getbuffer())
                jd_content = DocumentParser.parse_document(jd_path)["content"]
            else:
                jd_content = jd_text
            
            # Initialiser l'état du workflow
            initial_state = {
                "cv_text": cv_content,
                "cv_analysis": {},
                "jd_text": jd_content,
                "jd_analysis": {},
                "company_name": company_name,
                "company_info": {},
                "questions": [],
                "current_question_idx": 0,
                "user_answers": [],
                "feedback_history": [],
                "general_tips": {},
                "human_approval_needed": False,
                "human_feedback": "",
                "next_step": "",
                "error": ""
            }
            
            st.session_state.workflow_state = initial_state
            st.session_state.current_step = "analysis"
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur lors du parsing: {str(e)}")

def analysis_section():
    """Section d'analyse et génération de questions"""
    st.markdown('<div class="step-header"><h2>🔍 Étape 2: Analyse et Génération</h2></div>', 
                unsafe_allow_html=True)
    
    if st.session_state.workflow_state is None:
        st.warning("⚠️ Aucune donnée à analyser")
        return
    
    # Exécuter le workflow jusqu'au point de validation humaine
    with st.spinner("🤖 Les agents travaillent sur votre profil..."):
        try:
            supervisor = st.session_state.supervisor
            config = {"configurable": {"thread_id": "interview_prep_1"}}
            
            # Exécuter le workflow
            result = None
            for state in supervisor.graph.stream(st.session_state.workflow_state, config):
                result = state
                
                # Arrêter au point de validation humaine
                if list(result.values())[0].get("next_step") == "awaiting_human_input":
                    break
            
            if result:
                st.session_state.workflow_state = list(result.values())[0]
        
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return
    
    state = st.session_state.workflow_state
    
    # Afficher les résultats
    st.success("✅ Analyse terminée!")
    
    # Tabs pour organiser les résultats
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analyse CV", "📋 Analyse Poste", "🏢 Info Entreprise", "❓ Questions"])
    
    with tab1:
        if state.get("cv_analysis"):
            st.markdown("### Résumé de votre profil")
            cv_analysis = state["cv_analysis"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Points Forts:**")
                for strength in cv_analysis.get("strengths", []):
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("**📈 Axes d'Amélioration:**")
                for area in cv_analysis.get("areas_for_improvement", []):
                    st.markdown(f"- {area}")
            
            st.markdown("**💼 Compétences:**")
            skills = cv_analysis.get("skills", [])
            st.write(", ".join(skills) if skills else "Non spécifié")
            
            st.info(cv_analysis.get("summary", ""))
    
    with tab2:
        if state.get("jd_analysis"):
            jd_analysis = state["jd_analysis"]
            
            st.markdown(f"### {jd_analysis.get('job_title', 'Poste')}")
            st.markdown(f"**Niveau:** {jd_analysis.get('seniority_level', 'N/A')}")
            
            st.markdown("**🔧 Compétences Requises:**")
            for skill in jd_analysis.get("required_skills", []):
                st.markdown(f"- {skill}")
            
            st.markdown("**📝 Responsabilités Principales:**")
            for resp in jd_analysis.get("key_responsibilities", [])[:5]:
                st.markdown(f"- {resp}")
    
    with tab3:
        if state.get("company_info"):
            company_info = state["company_info"]
            
            st.markdown(f"### {company_info.get('company_name', 'Entreprise')}")
            st.markdown(f"**Activité:** {company_info.get('main_activity', 'N/A')}")
            
            st.markdown("**📰 Actualités Récentes:**")
            for news in company_info.get("recent_news", []):
                st.markdown(f"- {news}")
            
            st.markdown("**💡 Valeurs:**")
            values = company_info.get("values", [])
            st.write(", ".join(values) if values else "Non disponible")
    
    with tab4:
        if state.get("questions"):
            st.markdown("### Questions d'Entretien Générées")
            
            questions = state["questions"]
            
            # Grouper par catégorie
            categories = {}
            for q in questions:
                cat = q.get("category", "Autre")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(q)
            
            for category, questions_list in categories.items():
                with st.expander(f"📌 {category} ({len(questions_list)} questions)"):
                    for idx, q in enumerate(questions_list, 1):
                        st.markdown(f"**Q{idx}:** {q['question']}")
                        st.caption(f"🎯 Objectif: {q.get('objective', 'N/A')}")
                        
                        with st.container():
                            st.markdown("💡 **Conseils:**")
                            for tip in q.get("tips", []):
                                st.markdown(f"  - {tip}")
                        st.markdown("---")
    
    # Validation humaine
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 👤 Validation Humaine")
    st.write("Les questions vous conviennent-elles? Vous pouvez:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approuver et Continuer", type="primary", use_container_width=True):
            st.session_state.workflow_state["human_feedback"] = "approved"
            st.session_state.workflow_state["human_approval_needed"] = False
            st.session_state.current_step = "tips"
            st.rerun()
    
    with col2:
        if st.button("🔄 Régénérer Questions", use_container_width=True):
            st.session_state.workflow_state["human_feedback"] = "regenerate"
            st.rerun()
    
    with col3:
        if st.button("🎯 Passer à la Simulation", use_container_width=True):
            st.session_state.workflow_state["human_feedback"] = "interview"
            st.session_state.workflow_state["human_approval_needed"] = False
            st.session_state.current_step = "interview"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def tips_section():
    """Section de conseils de préparation"""
    st.markdown('<div class="step-header"><h2>💡 Étape 3: Conseils de Préparation</h2></div>', 
                unsafe_allow_html=True)
    
    state = st.session_state.workflow_state
    
    # Générer les conseils si pas encore fait
    if not state.get("general_tips"):
        with st.spinner("📝 Génération de conseils personnalisés..."):
            supervisor = st.session_state.supervisor
            config = {"configurable": {"thread_id": "interview_prep_1"}}
            
            # Continuer le workflow
            for result in supervisor.graph.stream(state, config):
                st.session_state.workflow_state = list(result.values())[0]
            
            state = st.session_state.workflow_state
    
    tips = state.get("general_tips", {})
    
    if tips:
        # Checklist de préparation
        st.markdown("### ✅ Checklist de Préparation")
        for item in tips.get("preparation_checklist", []):
            st.checkbox(item, key=f"checklist_{hash(item)}")
        
        # Points forts à mettre en avant
        with st.expander("💪 Points Forts à Mettre en Avant", expanded=True):
            for strength in tips.get("strengths_to_highlight", []):
                st.success(f"✓ {strength}")
        
        # Préoccupations potentielles
        with st.expander("⚠️ Points d'Attention"):
            concerns = tips.get("potential_concerns", [])
            for concern in concerns:
                if isinstance(concern, dict):
                    st.warning(f"**Préoccupation:** {concern.get('concern', '')}")
                    st.info(f"**Comment l'adresser:** {concern.get('how_to_address', '')}")
                else:
                    st.warning(concern)
        
        # Conseils pratiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👔 Code Vestimentaire")
            st.info(tips.get("dress_code", "Tenue professionnelle adaptée au secteur"))
            
            st.markdown("### 🚫 Erreurs à Éviter")
            for mistake in tips.get("common_mistakes", []):
                st.markdown(f"- {mistake}")
        
        with col2:
            st.markdown("### 🤝 Langage Corporel")
            st.info(tips.get("body_language", "Maintenez un contact visuel et une posture confiante"))
    
    # Bouton pour démarrer la simulation
    if st.button("🎭 Démarrer la Simulation d'Entretien", type="primary", use_container_width=True):
        st.session_state.current_step = "interview"
        st.session_state.interview_started = True
        st.rerun()

def interview_simulation_section():
    """Section de simulation d'entretien"""
    st.markdown('<div class="step-header"><h2>🎭 Étape 4: Simulation d\'Entretien</h2></div>',
                unsafe_allow_html=True)
    
    state = st.session_state.workflow_state
    questions = state.get("questions", [])
    current_idx = st.session_state.current_question
    
    if current_idx >= len(questions):
        # Entretien terminé
        st.success("🎉 Simulation d'entretien terminée!")
        st.session_state.current_step = "report"
        st.rerun()
        return
    
    # Barre de progression
    progress = current_idx / len(questions)
    st.progress(progress, text=f"Question {current_idx + 1} sur {len(questions)}")
    
    # Question courante
    current_question = questions[current_idx]
    
    st.markdown(f"### Question {current_idx + 1}")
    st.markdown(f"**Catégorie:** {current_question.get('category', 'N/A')}")
    st.markdown(f"**Difficulté:** {current_question.get('difficulty', 'medium').upper()}")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"## {current_question['question']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone de réponse
    answer = st.text_area(
        "Votre réponse:",
        height=200,
        key=f"answer_{current_idx}",
        placeholder="Prenez votre temps pour structurer une réponse claire et pertinente..."
    )
    
    # Conseils (optionnels)
    with st.expander("💡 Voir les conseils"):
        st.markdown("**Objectif de la question:**")
        st.write(current_question.get('objective', 'N/A'))
        
        st.markdown("**Conseils pour répondre:**")
        for tip in current_question.get('tips', []):
            st.markdown(f"- {tip}")
    
    # Boutons d'action
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("📝 Soumettre et Obtenir Feedback", type="primary", use_container_width=True):
            if not answer or len(answer.strip()) < 10:
                st.error("⚠️ Veuillez fournir une réponse plus détaillée")
            else:
                # Sauvegarder la réponse
                if "user_answers" not in state:
                    state["user_answers"] = []
                
                state["user_answers"].append({
                    "question_idx": current_idx,
                    "question": current_question["question"],
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Obtenir le feedback
                with st.spinner("🤖 Analyse de votre réponse..."):
                    coach = st.session_state.supervisor.agents["interview_coach"]
                    context = {
                        "cv": state["cv_analysis"],
                        "jd": state["jd_analysis"]
                    }
                    
                    feedback_result = coach.evaluate_answer(
                        current_question["question"],
                        answer,
                        context
                    )
                    
                    if feedback_result["success"]:
                        feedback = feedback_result["feedback"]
                        
                        # Afficher le feedback
                        st.markdown("---")
                        st.markdown("### 📊 Feedback sur Votre Réponse")
                        
                        # Score
                        score = feedback.get("score", 0)
                        st.metric("Score", f"{score}/10")
                        
                        # Points positifs
                        st.markdown("#### ✅ Points Positifs")
                        for point in feedback.get("positive_points", []):
                            st.success(point)
                        
                        # Points à améliorer
                        st.markdown("#### 📈 Points à Améliorer")
                        for point in feedback.get("improvement_areas", []):
                            st.warning(point)
                        
                        # Réponse améliorée
                        with st.expander("💡 Suggestion de Réponse Améliorée"):
                            st.write(feedback.get("improved_answer", ""))
                        
                        # Conseils spécifiques
                        st.markdown("#### 🎯 Conseils Spécifiques")
                        for tip in feedback.get("specific_tips", []):
                            st.info(tip)
                        
                        # Encouragement
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"**💪 {feedback.get('encouragement', 'Continuez comme ça!')}**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Sauvegarder le feedback
                        if "feedback_history" not in state:
                            state["feedback_history"] = []
                        
                        state["feedback_history"].append({
                            "question_idx": current_idx,
                            "feedback": feedback
                        })
                        
                        st.session_state.workflow_state = state
    
    with col2:
        if st.button("⏭️ Question Suivante", use_container_width=True):
            st.session_state.current_question += 1
            st.rerun()
    
    with col3:
        if st.button("⏸️ Pause", use_container_width=True):
            st.info("Simulation en pause. Cliquez sur 'Continuer' quand vous êtes prêt.")

def report_section():
    """Section de rapport final"""
    st.markdown('<div class="step-header"><h2>📊 Rapport Final de Préparation</h2></div>', 
                unsafe_allow_html=True)
    
    state = st.session_state.workflow_state
    
    st.balloons()
    st.success("🎉 Félicitations! Vous avez terminé votre préparation d'entretien!")
    
    # Statistiques globales
    st.markdown("### 📈 Vos Statistiques")
    
    col1, col2, col3, col4 = st.columns(4)
    
    feedback_history = state.get("feedback_history", [])
    
    if feedback_history:
        avg_score = sum(f["feedback"].get("score", 0) for f in feedback_history) / len(feedback_history)
        
        with col1:
            st.metric("Score Moyen", f"{avg_score:.1f}/10")
        
        with col2:
            st.metric("Questions Répondues", len(feedback_history))
        
        with col3:
            good_scores = sum(1 for f in feedback_history if f["feedback"].get("score", 0) >= 7)
            st.metric("Bonnes Réponses", f"{good_scores}/{len(feedback_history)}")
        
        with col4:
            completion = (len(feedback_history) / len(state.get("questions", []))) * 100
            st.metric("Complétion", f"{completion:.0f}%")
    
    # Détails par question
    st.markdown("### 📝 Détail de Vos Réponses")
    
    for idx, feedback_item in enumerate(feedback_history, 1):
        with st.expander(f"Question {idx} - Score: {feedback_item['feedback'].get('score', 0)}/10"):
            user_answer = next((a for a in state.get("user_answers", []) 
                              if a["question_idx"] == feedback_item["question_idx"]), None)
            
            if user_answer:
                st.markdown(f"**Question:** {user_answer['question']}")
                st.markdown(f"**Votre réponse:** {user_answer['answer']}")
                
                feedback = feedback_item["feedback"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Points Positifs:**")
                    for point in feedback.get("positive_points", []):
                        st.markdown(f"- ✅ {point}")
                
                with col2:
                    st.markdown("**Points à Améliorer:**")
                    for point in feedback.get("improvement_areas", []):
                        st.markdown(f"- 📈 {point}")
    
    # Recommandations finales
    st.markdown("### 🎯 Recommandations Finales")
    
    if feedback_history:
        # Analyser les forces et faiblesses
        all_improvement_areas = []
        all_positive_points = []
        
        for f in feedback_history:
            all_improvement_areas.extend(f["feedback"].get("improvement_areas", []))
            all_positive_points.extend(f["feedback"].get("positive_points", []))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💪 Vos Forces")
            unique_strengths = list(set(all_positive_points))[:5]
            for strength in unique_strengths:
                st.success(f"✓ {strength}")
        
        with col2:
            st.markdown("#### 📚 Axes de Travail")
            unique_improvements = list(set(all_improvement_areas))[:5]
            for improvement in unique_improvements:
                st.warning(f"→ {improvement}")
    
    # Export du rapport
    st.markdown("### 💾 Export du Rapport")
    
    if st.button("📄 Télécharger le Rapport (JSON)", use_container_width=True):
        report_data = {
            "date": datetime.now().isoformat(),
            "company": state.get("company_name", "N/A"),
            "position": state.get("jd_analysis", {}).get("job_title", "N/A"),
            "statistics": {
                "average_score": avg_score if feedback_history else 0,
                "questions_answered": len(feedback_history),
                "total_questions": len(state.get("questions", []))
            },
            "answers": state.get("user_answers", []),
            "feedback": feedback_history,
            "tips": state.get("general_tips", {})
        }
        
        json_str = json.dumps(report_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Télécharger",
            data=json_str,
            file_name=f"interview_prep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Bouton pour recommencer
    if st.button("🔄 Nouvelle Préparation", type="primary", use_container_width=True):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    """Fonction principale de l'application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 InterviewMaster AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Votre coach IA pour réussir vos entretiens d\'embauche</p>', unsafe_allow_html=True)
    
    # Initialiser la session
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        # Indicateur de progression
        steps = ["upload", "analysis", "tips", "interview", "report"]
        current_step_idx = steps.index(st.session_state.current_step) if st.session_state.current_step in steps else 0
        
        for idx, step in enumerate(steps):
            step_names = {
                "upload": "1️⃣ Upload Documents",
                "analysis": "2️⃣ Analyse & Questions",
                "tips": "3️⃣ Conseils",
                "interview": "4️⃣ Simulation",
                "report": "5️⃣ Rapport"
            }
            
            if idx < current_step_idx:
                st.success(step_names[step] + " ✅")
            elif idx == current_step_idx:
                st.info(step_names[step] + " 🔄")
            else:
                st.text(step_names[step])
        
        st.markdown("---")
        
        # Informations
        st.markdown("## ℹ️ À Propos")
        st.markdown("""
        **InterviewMaster AI** utilise:
        - 🤖 Agents IA multi-spécialisés
        - 🔍 Recherche web en temps réel
        - 📊 RAG pour analyse contextuelle
        - 💾 Persistance des sessions
        - 📈 Monitoring Langfuse
        """)
        
        st.markdown("---")
        
        # Aide
        with st.expander("❓ Aide"):
            st.markdown("""
            **Comment utiliser l'application:**
            
            1. **Upload**: Fournissez votre CV et la description de poste
            2. **Analyse**: Laissez les agents analyser et générer des questions
            3. **Validation**: Approuvez ou régénérez les questions
            4. **Conseils**: Consultez les recommandations personnalisées
            5. **Simulation**: Répondez aux questions et recevez du feedback
            6. **Rapport**: Consultez vos statistiques et téléchargez le rapport
            """)
        
        # Status des agents
        if st.session_state.agents_initialized:
            st.success("🟢 Agents IA: Actifs")
        else:
            st.warning("🟡 Agents IA: Non initialisés")
    
    # Initialiser les agents
    initialize_agents()
    
    # Router vers la bonne section
    if st.session_state.current_step == "upload":
        upload_documents_section()
    elif st.session_state.current_step == "analysis":
        analysis_section()
    elif st.session_state.current_step == "tips":
        tips_section()
    elif st.session_state.current_step == "interview":
        interview_simulation_section()
    elif st.session_state.current_step == "report":
        report_section()

if __name__ == "__main__":
    main()