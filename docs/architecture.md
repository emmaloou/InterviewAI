# Architecture Technique - InterviewMaster AI

## Vue d'ensemble

InterviewMaster AI est un système multi-agents orchestré par LangGraph qui prépare les candidats aux entretiens d'embauche.

## Composants Principaux

### 1. Interface Utilisateur (Streamlit)
- Upload de documents
- Visualisation des analyses
- Simulation interactive
- Rapport final

### 2. Orchestrateur (LangGraph Supervisor)
- Gestion du workflow
- Routing entre agents
- Human-in-the-loop
- Persistance des états

### 3. Agents Spécialisés

#### CV Analyzer
- **Input**: Texte du CV
- **Output**: Compétences, expérience, points forts/faibles
- **LLM**: Analyse structurée en JSON

#### JD Analyzer
- **Input**: Description de poste
- **Output**: Exigences, responsabilités, culture
- **LLM**: Extraction d'entités

#### Company Researcher
- **Input**: Nom d'entreprise
- **Tools**: Web search (Tavily)
- **Output**: Informations récentes, culture, valeurs

#### Question Generator
- **Input**: CV + JD + Info entreprise
- **Output**: 10 questions catégorisées avec conseils
- **LLM**: Génération contextuelle

#### Interview Coach
- **Input**: Question + Réponse candidat
- **Output**: Score, feedback, suggestions
- **LLM**: Évaluation qualitative

### 4. Outils

#### Document Parser
- Supporte PDF, DOCX, TXT
- Extraction de texte brut

#### Web Search Tool
- API Tavily
- Recherche entreprise et conseils

#### Vector Store
- ChromaDB
- RAG sur documents
- Recherche sémantique

### 5. Monitoring
- Langfuse pour observabilité
- Traces complètes
- Métriques de performance

## Flux de Données
```
1. User uploads CV + JD → Document Parser
2. Supervisor → CV Analyzer → Analysis stored in Vector DB
3. Supervisor → JD Analyzer → Analysis stored in Vector DB
4. Supervisor → Company Researcher → Web Search → Synthesis
5. Supervisor → Question Generator → 10 questions
6. Human Review → Approve/Regenerate
7. Supervisor → Tips Generator → Personalized advice
8. Interview Loop:
   - Display question
   - User answers
   - Coach evaluates → Feedback
   - Next question
9. Final Report → Statistics + Export
```

## Décisions Techniques

### Pourquoi LangGraph?
- Orchestration complexe multi-agents
- Support natif human-in-the-loop
- Persistance avec checkpoints
- Routing conditionnel

### Pourquoi Streamlit?
- Développement rapide
- Déploiement facile (Streamlit Cloud)
- Interface réactive
- Support file upload

### Pourquoi ChromaDB?
- Léger et embarquable
- Pas de serveur requis
- Support embeddings
- Bon pour prototypes

## Limitations Connues

1. **Latence**: Appels LLM séquentiels (10-30s)
2. **Parsing**: OCR limité sur PDFs scannés
3. **Multilingue**: Optimisé pour français/anglais
4. **Scale**: Conçu pour usage individuel

## Améliorations Futures

1. Cache des analyses
2. Parallélisation des agents
3. Support vidéo (simulation interview)
4. Multi-utilisateurs avec auth
5. Analytics dashboard