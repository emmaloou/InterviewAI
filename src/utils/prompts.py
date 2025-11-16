# Templates de prompts pour chaque agent
CV_ANALYSIS_PROMPT = """
Tu es un expert en analyse de CV. Analyse le CV suivant et extrais:
1. Compétences principales
2. Expérience (années et domaines)
3. Formation
4. Points forts
5. Points à améliorer

CV:
{cv_text}

Fournis une analyse structurée en JSON.
"""

JD_ANALYSIS_PROMPT = """
Tu es un expert RH. Analyse la description de poste suivante:
{jd_text}

Extrais:
1. Poste et niveau
2. Compétences requises
3. Expérience demandée
4. Responsabilités clés
5. Culture d'entreprise

Format JSON.
"""

COMPANY_RESEARCH_PROMPT = """
Recherche des informations sur l'entreprise: {company_name}
Domaine: {industry}

Trouve:
1. Activité principale
2. Actualités récentes
3. Culture d'entreprise
4. Valeurs
5. Défis du secteur

Utilise les informations de recherche web fournies.
"""

QUESTION_GENERATION_PROMPT = """
Génère 10 questions d'entretien pertinentes basées sur:

Profil candidat:
{cv_summary}

Poste visé:
{jd_summary}

Informations entreprise:
{company_info}

Catégories:
- Questions techniques (3)
- Questions comportementales (3)
- Questions sur l'entreprise (2)
- Questions de mise en situation (2)

Format: JSON avec catégorie, question, objectif, conseils de réponse
"""

FEEDBACK_PROMPT = """
Évalue la réponse du candidat à la question d'entretien:

Question: {question}
Réponse du candidat: {answer}
Contexte (CV/JD): {context}

Fournis:
1. Note sur 10
2. Points positifs
3. Points à améliorer
4. Suggestion de réponse améliorée

Sois constructif et encourageant.
"""