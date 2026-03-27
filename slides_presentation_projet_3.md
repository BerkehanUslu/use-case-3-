# Slides — Projet 3

## Slide 1 — Titre

**Assistant Intelligent d'Analyse Immobilière avec Hugging Face**

- Analyse du marché immobilier français
- Données DVF + BAN + textes/PDF
- NLP + API + interface utilisateur

À dire :
"Je vais présenter un assistant intelligent capable d'analyser des données immobilières françaises et de répondre à des besoins concrets comme le prix d'une commune, le résumé de rapports ou l'analyse d'une annonce."

---

## Slide 2 — Objectif

- Répondre à des questions sur l'immobilier français
- Résumer des documents immobiliers
- Analyser des descriptions de biens
- Fournir une interface simple à utiliser

À dire :
"L'objectif est de créer un outil unique qui transforme des données et des textes immobiliers en informations directement utiles."

---

## Slide 3 — Données utilisées

- DVF : transactions immobilières
- BAN : coordonnées géographiques
- Textes et PDF : rapports et annonces

À dire :
"Le projet combine des données structurées et des données textuelles. Cela le rend plus riche qu'un simple tableau de prix."

---

## Slide 4 — Pipeline global

- Préparation des données
- Indexation vectorielle avec FAISS
- Modèles Hugging Face
- API FastAPI
- Interface Streamlit

À dire :
"Le projet suit une logique de pipeline, de la donnée brute jusqu'à l'application finale visible par l'utilisateur."

---

## Slide 5 — Préparation des données DVF

- Lecture des fichiers DVF
- Nettoyage des données
- Calcul du prix au m²
- Agrégation par commune, année et type de bien

Fichier :
`backend/01_data_preparation.py`

À dire :
"Cette étape transforme les données brutes en indicateurs simples comme le prix moyen au mètre carré."

---

## Slide 6 — Recherche sémantique avec FAISS

- Création de descriptions de communes
- Transformation en vecteurs
- Indexation avec FAISS
- Recherche par similarité

Fichier :
`backend/02_vector_indexing.py`

À dire :
"FAISS permet de retrouver des informations proches du sens d'une requête, pas seulement des mots exacts."

---

## Slide 7 — Modèles Hugging Face

- CamemBERT : question-réponse
- BARThez : résumé automatique
- BERT multilingual : sentiment
- Sentence-Transformers : embeddings

À dire :
"Chaque modèle a un rôle précis. Le projet repose donc sur plusieurs briques NLP spécialisées."

---

## Slide 8 — API et Interface

- FastAPI expose les fonctionnalités
- Streamlit fournit une interface simple
- 5 onglets utilisateur

Fichiers :
- `backend/06_api_deployment.py`
- `backend/07_streamlit_app.py`

À dire :
"FastAPI relie les modèles à l'interface, et Streamlit rend le tout facile à démontrer."

---

## Slide 9 — Fonctionnalités visibles

- Prix par commune
- Questions-réponses
- Résumé de rapport
- Analyse de bien
- Carte interactive

À dire :
"L'utilisateur interagit avec plusieurs modules spécialisés, mais tout est réuni dans une seule application."

---

## Slide 10 — Forces et limites

Forces :

- projet complet de bout en bout
- utilisation de données réelles
- application concrète

Limites :

- modèles lourds
- dépendance à la qualité des données
- recommandations simplifiées

À dire :
"Le projet est solide parce qu'il va de la donnée à l'interface. Mais il garde des limites classiques des systèmes IA, notamment la dépendance aux données et au contexte."

---

## Slide 11 — Conclusion

- Données publiques + IA + interface
- Projet modulaire et démontrable
- Cas d'usage concret pour l'immobilier

À dire :
"Ce projet montre comment construire un assistant immobilier intelligent complet en combinant traitement de données, NLP et visualisation."
