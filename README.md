# 🎯 InterviewMaster AI

Assistant IA multi-agents pour la préparation d'entretiens d'embauche avec analyse de CV, recherche entreprise, génération de questions et simulation interactive.

## 🚀 Fonctionnalités

- **Analyse de CV**: Extraction automatique des compétences, expériences et points forts
- **Analyse de Poste**: Compréhension des exigences et responsabilités
- **Recherche Entreprise**: Information en temps réel via recherche web
- **Génération de Questions**: Questions personnalisées par catégorie
- **Simulation Interactive**: Mode interview avec feedback en temps réel
- **Monitoring Langfuse**: Observabilité complète du workflow
- **Persistance**: Sauvegarde de sessions avec SqliteSaver

## 🏗️ Architecture
```
┌─────────────────────┐
│   Streamlit UI      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  LangGraph Workflow │
│    (Supervisor)     │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │   Agents    │
    ├─────────────┤
    │ CV Analyzer │
    │ JD Analyzer │
    │ Researcher  │
    │ Q Generator │
    │ Coach       │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │   Tools     │
    ├─────────────┤
    │ Vector DB   │
    │ Web Search  │
    │ Doc Parser  │
    └─────────────┘
```

## 📋 Prérequis

- Python 3.10+
- Ollama (pour LLM local) OU clés API (OpenAI, Groq, etc.)
- Compte Tavily (API search gratuite)
- Compte Langfuse (monitoring)

## 🔧 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/votre-username/interview-master-ai.git
cd interview-master-ai
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration

Copier `.env.example` vers `.env` et remplir les variables:
```bash
cp .env.example .env
```

Éditer `.env`:
```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b

# Tavily
TAVILY_API_KEY=votre_clé

# Langfuse
LANGFUSE_PUBLIC_KEY=votre_clé
LANGFUSE_SECRET_KEY=votre_clé
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Installer et lancer Ollama (si local)
```bash
# Télécharger depuis https://ollama.com
ollama pull llama3.1:8b
ollama serve
```

## 🎮 Utilisation

### Lancer l'application
```bash
streamlit run src/ui/streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

### Workflow

1. **Upload** votre CV (PDF, DOCX, TXT)
2. **Fournir** la description de poste
3. **Entrer** le nom de l'entreprise
4. **Lancer l'analyse** et attendre la génération
5. **Valider** les questions proposées
6. **Consulter** les conseils personnalisés
7. **Simuler** l'entretien avec feedback temps réel
8. **Télécharger** le rapport final

## 🧪 Tests
```bash
# Tests unitaires
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

## 📊 Monitoring Langfuse

Accédez à votre dashboard Langfuse pour voir:
- Traces des exécutions d'agents
- Temps de réponse
- Tokens utilisés
- Erreurs et exceptions
- Métriques de qualité

## 🚀 Déploiement

### Streamlit Cloud

1. Push le code sur GitHub
2. Connectez-vous sur [streamlit.io](https://streamlit.io)
3. Déployez depuis le repository
4. Ajoutez les secrets dans Settings

### Hugging Face Spaces
```bash
# Créer un Space sur HF
# Ajouter un fichier app.py à la racine:
```

**app.py:**
```python
from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()
```

### Docker (optionnel)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t interview-master-ai .
docker run -p 8501:8501 --env-file .env interview-master-ai
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

MIT License

## 👥 Auteurs

Votre Équipe

## 🙏 Remerciements

- LangChain & LangGraph
- Anthropic Claude
- Streamlit
- Tavily
- Langfuse