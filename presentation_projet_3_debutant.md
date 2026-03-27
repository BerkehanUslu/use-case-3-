# Présentation Simple — Projet 3

## Assistant Intelligent d'Analyse Immobilière avec Hugging Face

Ce document est fait pour un débutant complet.
Le but est double :

1. comprendre le projet étape par étape ;
2. avoir un support simple à présenter au professeur.

---

## 1. Résumé ultra simple

### Le projet en une phrase

Ce projet construit un assistant immobilier capable de répondre à des questions sur le marché immobilier français en combinant :

- des données immobilières réelles ;
- des modèles NLP de Hugging Face ;
- une API ;
- une interface Streamlit.

### Ce que je peux dire à l'oral

"L'idée du projet est de créer un assistant intelligent qui comprend des questions en français sur l'immobilier, analyse des données de vente, résume des textes, et aide l'utilisateur à explorer le marché immobilier."

---

## 2. Le problème à résoudre

### Pourquoi ce projet existe

Un professionnel de l'immobilier peut avoir besoin de :

- connaître le prix moyen dans une commune ;
- voir l'évolution des prix ;
- résumer un rapport immobilier long ;
- analyser une description de bien ;
- retrouver rapidement une information pertinente.

### Ce que le projet apporte

Au lieu de lire des tableaux ou des documents manuellement, l'utilisateur pose une question ou colle un texte, et l'assistant répond.

### Ce que je peux dire à l'oral

"Le projet sert à gagner du temps. Au lieu de chercher l'information dans plusieurs fichiers, l'utilisateur interroge un assistant qui analyse les données et les textes automatiquement."

---

## 3. Les données utilisées

### Source 1 : DVF

DVF veut dire `Demandes de Valeurs Foncières`.

Ce sont des données publiques françaises sur les transactions immobilières.

Le script [backend/01_data_preparation.py](/Users/berkehan/Desktop/use-case-3-/backend/01_data_preparation.py) lit ces fichiers bruts et calcule notamment :

- le prix moyen au m² ;
- le prix médian au m² ;
- le nombre de transactions ;
- la commune ;
- l'année ;
- le type de bien.

### Source 2 : BAN

BAN veut dire `Base Adresse Nationale`.

Elle sert ici à récupérer des coordonnées géographiques pour les communes.

Cela permet d'afficher les données sur une carte.

Cette partie est gérée dans [backend/02_vector_indexing.py](/Users/berkehan/Desktop/use-case-3-/backend/02_vector_indexing.py).

### Source 3 : textes et PDF

Le projet peut aussi traiter :

- des descriptions de biens ;
- des rapports immobiliers ;
- des fichiers PDF.

### Ce que je peux dire à l'oral

"Le projet mélange des données structurées comme les ventes DVF, des coordonnées géographiques avec BAN, et des textes libres comme des annonces ou des rapports PDF."

---

## 4. Architecture globale

### Vue très simple

Le projet suit cette chaîne :

1. on prépare les données DVF ;
2. on construit un index de recherche sémantique ;
3. on charge plusieurs modèles Hugging Face ;
4. on expose tout cela dans une API FastAPI ;
5. on utilise une interface Streamlit pour l'utilisateur.

### Pipeline réel du projet

`Données DVF -> agrégation -> index FAISS -> modèles NLP -> API -> interface utilisateur`

### Ce que je peux dire à l'oral

"Le projet est organisé comme une chaîne de traitement. D'abord on prépare les données, ensuite on les rend interrogeables, puis on branche des modèles de langage, et enfin on donne accès à tout cela avec une API et une interface graphique."

---

## 5. Étape 1 — Préparation des données

### Fichier principal

[backend/01_data_preparation.py](/Users/berkehan/Desktop/use-case-3-/backend/01_data_preparation.py)

### Ce que fait ce script

Il lit les fichiers DVF bruts, puis :

1. vérifie que les colonnes importantes existent ;
2. nettoie les valeurs manquantes ;
3. convertit les prix en nombres ;
4. calcule le prix au m² ;
5. regroupe les données par commune, année et type de bien ;
6. sauvegarde le résultat dans un fichier Parquet.

### Pourquoi cette étape est importante

Les données brutes sont trop volumineuses et trop désordonnées pour être utilisées directement.

Cette étape transforme les données en une version plus claire et plus rapide à exploiter.

### Sortie produite

`backend/data/dvf_communes.parquet`

### Ce que je peux dire à l'oral

"La première étape consiste à transformer les données brutes DVF en indicateurs utiles, par exemple le prix moyen au mètre carré par commune et par année."

---

## 6. Étape 2 — Création de l'index vectoriel

### Fichier principal

[backend/02_vector_indexing.py](/Users/berkehan/Desktop/use-case-3-/backend/02_vector_indexing.py)

### Idée simple

Un ordinateur comprend mal directement le texte.
On transforme donc chaque description en vecteur numérique.

Ensuite, FAISS permet de retrouver rapidement les éléments les plus proches d'une requête.

### Ce que fait ce script

1. lit les données agrégées ;
2. crée une phrase descriptive pour chaque commune ;
3. transforme ces phrases en vecteurs avec `sentence-transformers` ;
4. stocke les vecteurs dans un index FAISS ;
5. ajoute si possible les coordonnées géographiques grâce à BAN ;
6. sauvegarde l'index et les métadonnées.

### Fichiers produits

- `backend/indexes/real_estate.faiss`
- `backend/indexes/real_estate_meta.json`

### À quoi cela sert

Si l'utilisateur écrit par exemple :

`prix immobilier Paris appartement`

le système peut retrouver les communes ou entrées les plus pertinentes, même si les mots ne sont pas exactement identiques.

### Ce que je peux dire à l'oral

"FAISS sert à faire de la recherche intelligente. On transforme le texte en vecteurs, puis on retrouve les informations proches du sens de la question."

---

## 7. Étape 3 — Question / Réponse avec Hugging Face

### Fichier principal

[backend/03_qa_system.py](/Users/berkehan/Desktop/use-case-3-/backend/03_qa_system.py)

### Modèle utilisé

Le projet utilise un modèle CamemBERT français pour le question-answering.

### Comment cela fonctionne

1. l'utilisateur pose une question ;
2. le système cherche un contexte utile ;
3. le modèle lit la question + le contexte ;
4. il extrait une réponse dans le texte.

### Important à comprendre

Ce n'est pas un chatbot qui invente librement une réponse.

Ici, le modèle cherche surtout une réponse dans un texte fourni ou récupéré.
On parle donc de `question answering extractif`.

### Mode RAG simplifié

Si aucun contexte n'est donné par l'utilisateur, le script reconstruit un contexte à partir des métadonnées des communes.

### Ce que je peux dire à l'oral

"Le système de questions-réponses ne génère pas une réponse au hasard. Il cherche un contexte pertinent puis extrait la réponse la plus probable à partir de ce contexte."

---

## 8. Étape 4 — Résumé automatique

### Fichier principal

[backend/04_summarization.py](/Users/berkehan/Desktop/use-case-3-/backend/04_summarization.py)

### Modèle utilisé

Le projet utilise BARThez pour résumer du texte en français.

### Ce que fait ce module

Il peut :

- résumer un texte ;
- résumer un PDF ;
- découper un long document en morceaux ;
- résumer chaque morceau ;
- produire un résumé final.

### Pourquoi c'est utile

Dans l'immobilier, les rapports et études sont souvent longs.
Le résumé automatique permet d'obtenir rapidement l'idée principale.

### Ce que je peux dire à l'oral

"Le module de résumé sert à condenser un rapport ou une annonce longue. Quand le texte est trop long, le système le coupe en morceaux avant de produire un résumé final."

---

## 9. Étape 5 — Analyse de sentiment

### Fichier principal

[backend/05_sentiment_analysis.py](/Users/berkehan/Desktop/use-case-3-/backend/05_sentiment_analysis.py)

### Rôle de ce module

Il analyse le ton d'une description de bien :

- très positif ;
- positif ;
- neutre ;
- négatif ;
- très négatif.

### Exemple

Une annonce comme :

`Magnifique appartement lumineux, rénové, proche transports`

sera probablement détectée comme positive.

### Ce que fait ensuite le projet

Le score de sentiment est converti en recommandation d'investissement simple.

### Limite importante

Cette recommandation n'est pas une expertise financière complète.
C'est surtout une aide d'interprétation basée sur le texte et parfois sur le prix.

### Ce que je peux dire à l'oral

"L'analyse de sentiment permet d'évaluer si la description d'un bien semble attractive ou risquée. Ensuite, le système transforme ce résultat en recommandation simple pour l'utilisateur."

---

## 10. Étape 6 — API FastAPI

### Fichier principal

[backend/06_api_deployment.py](/Users/berkehan/Desktop/use-case-3-/backend/06_api_deployment.py)

### Pourquoi une API

Une API permet à d'autres applications de communiquer avec le projet.

Ici, l'API sert de pont entre les modèles Python et l'interface utilisateur.

### Endpoints principaux

- `/health` : vérifier que l'application fonctionne ;
- `/communes` : récupérer la liste des communes ;
- `/communes/{commune}` : obtenir les données d'une commune ;
- `/trends/{commune}` : voir l'évolution des prix ;
- `/qa` : poser une question ;
- `/summarize` : résumer un texte ;
- `/summarize-pdf` : résumer un PDF ;
- `/sentiment` : analyser une description ;
- `/search` : faire une recherche sémantique.

### Ce que je peux dire à l'oral

"FastAPI transforme les scripts en services accessibles par requêtes HTTP. Cela rend le projet plus propre, modulaire et réutilisable."

---

## 11. Étape 7 — Interface Streamlit

### Fichier principal

[backend/07_streamlit_app.py](/Users/berkehan/Desktop/use-case-3-/backend/07_streamlit_app.py)

### Rôle de l'interface

C'est la partie visible par l'utilisateur final.

Elle contient 5 onglets :

1. prix par commune ;
2. Q&A immobilier ;
3. résumé de rapport ;
4. analyse de bien ;
5. carte interactive.

### Ce que voit l'utilisateur

L'utilisateur n'a pas besoin de lancer les modèles à la main.
Il remplit un champ, clique sur un bouton, puis voit le résultat.

### Détail intéressant

L'application peut fonctionner :

- en mode API normal ;
- en mode démonstration si l'API n'est pas disponible.

### Ce que je peux dire à l'oral

"Streamlit rend le projet concret et démontrable. C'est grâce à cette interface que l'assistant devient facile à utiliser, même pour une personne non technique."

---

## 12. Les modèles Hugging Face utilisés

### 1. Embeddings

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Rôle :
transformer du texte en vecteurs pour la recherche sémantique.

### 2. Question-réponse

`etalab-ia/camembert-base-squadFR-fquad-piaf`

Rôle :
répondre à une question à partir d'un contexte en français.

### 3. Résumé

`moussaKam/barthez-orangesum-abstract`

Rôle :
résumer des textes et rapports en français.

### 4. Sentiment

`nlptown/bert-base-multilingual-uncased-sentiment`

Rôle :
attribuer un score de sentiment à une description.

### Ce que je peux dire à l'oral

"Le projet n'utilise pas un seul modèle, mais plusieurs modèles spécialisés. Chaque modèle a une tâche précise : rechercher, répondre, résumer ou analyser le ton d'un texte."

---

## 13. Exemple de parcours utilisateur

### Cas 1 : prix d'une commune

1. l'utilisateur tape `Paris` ;
2. l'interface appelle l'API ;
3. l'API lit les données agrégées ;
4. elle renvoie les prix et le nombre de transactions ;
5. Streamlit affiche les métriques et les graphiques.

### Cas 2 : question immobilière

1. l'utilisateur pose une question ;
2. le backend cherche un contexte utile ;
3. le modèle QA produit une réponse ;
4. le résultat est affiché avec un score de confiance.

### Cas 3 : analyse d'une annonce

1. l'utilisateur colle une description ;
2. le backend exécute l'analyse de sentiment ;
3. le système retourne un score ;
4. l'interface affiche une recommandation visuelle.

### Ce que je peux dire à l'oral

"Le projet a été pensé comme un assistant pratique. Chaque action de l'utilisateur déclenche un module spécialisé en arrière-plan, puis l'interface affiche une réponse compréhensible."

---

## 14. Ce que le projet réussit bien

- il combine données structurées et NLP ;
- il couvre plusieurs usages réels ;
- il est modulaire ;
- il possède une API et une interface ;
- il est facilement démontrable ;
- il utilise des données publiques françaises.

### Ce que je peux dire à l'oral

"Le point fort du projet est l'intégration complète : préparation des données, intelligence artificielle, API et interface utilisateur dans une seule application."

---

## 15. Limites du projet

Il faut être honnête sur les limites.

### Limites techniques

- les modèles Hugging Face peuvent être lourds à charger ;
- certaines réponses dépendent de la qualité du contexte ;
- l'analyse de sentiment reste simple ;
- la recommandation d'investissement n'est pas un conseil expert ;
- sans internet ou sans modèles téléchargés, certaines fonctions peuvent être limitées.

### Ce que je peux dire à l'oral

"Comme tout projet IA, il y a des limites. Les modèles ne remplacent pas un expert humain, et la qualité des résultats dépend beaucoup des données et du contexte utilisé."

---

## 16. Conclusion simple

### Bilan

Ce projet montre comment construire un assistant immobilier intelligent de bout en bout :

- préparation de données DVF ;
- recherche sémantique avec FAISS ;
- NLP avec Hugging Face ;
- API FastAPI ;
- interface Streamlit.

### Phrase finale pour l'oral

"En conclusion, ce projet illustre une chaîne complète d'IA appliquée à l'immobilier français. Il transforme des données brutes et des textes en réponses utiles pour un utilisateur final."

---

## 17. Version courte pour parler 2 à 3 minutes

Vous pouvez presque lire ce texte tel quel :

"Bonjour, je vais vous présenter mon projet d'assistant intelligent d'analyse immobilière. L'objectif est de créer un outil capable de répondre à des questions sur le marché immobilier français en utilisant les données DVF et plusieurs modèles Hugging Face. D'abord, le projet nettoie et agrège les données de transactions immobilières pour obtenir des indicateurs comme le prix moyen au mètre carré par commune. Ensuite, il transforme ces informations en vecteurs grâce à un modèle d'embeddings, puis les stocke dans FAISS pour permettre une recherche sémantique rapide. Le projet contient ensuite plusieurs modules NLP : un module de question-réponse avec CamemBERT, un module de résumé automatique avec BARThez, et un module d'analyse de sentiment pour étudier les descriptions de biens. Tous ces modules sont reliés par une API FastAPI, puis présentés dans une interface Streamlit avec plusieurs onglets. L'intérêt de ce projet est qu'il combine données publiques, intelligence artificielle et interface utilisateur dans une seule application concrète. Enfin, il montre qu'on peut transformer des données immobilières brutes en assistant intelligent capable d'aider à l'analyse et à la décision."

---

## 18. Questions possibles du professeur

### Question : pourquoi utiliser FAISS ?

Réponse simple :

"FAISS permet de faire de la recherche sémantique rapide sur des vecteurs. Cela aide à retrouver les informations proches du sens d'une requête, pas seulement des mots exacts."

### Question : pourquoi Hugging Face ?

Réponse simple :

"Hugging Face fournit des modèles déjà entraînés pour des tâches NLP comme le résumé, le question-answering ou le sentiment, ce qui accélère fortement le développement."

### Question : pourquoi FastAPI ?

Réponse simple :

"FastAPI permet d'exposer les fonctions du projet comme une API propre, rapide et documentée."

### Question : pourquoi Streamlit ?

Réponse simple :

"Streamlit permet de créer rapidement une interface de démonstration pour tester le projet sans développer un front-end complexe."

### Question : quelle est la vraie valeur du projet ?

Réponse simple :

"La vraie valeur est d'unifier des données immobilières, des modèles NLP et une interface simple dans un seul assistant exploitable."

---

## 19. Une phrase pour chaque fichier principal

### [backend/01_data_preparation.py](/Users/berkehan/Desktop/use-case-3-/backend/01_data_preparation.py)

Prépare et agrège les données DVF.

### [backend/02_vector_indexing.py](/Users/berkehan/Desktop/use-case-3-/backend/02_vector_indexing.py)

Construit l'index FAISS et les métadonnées de recherche.

### [backend/03_qa_system.py](/Users/berkehan/Desktop/use-case-3-/backend/03_qa_system.py)

Répond aux questions en français à partir d'un contexte.

### [backend/04_summarization.py](/Users/berkehan/Desktop/use-case-3-/backend/04_summarization.py)

Résume des textes et des PDF.

### [backend/05_sentiment_analysis.py](/Users/berkehan/Desktop/use-case-3-/backend/05_sentiment_analysis.py)

Analyse le ton d'une description de bien.

### [backend/06_api_deployment.py](/Users/berkehan/Desktop/use-case-3-/backend/06_api_deployment.py)

Expose toutes les fonctions sous forme d'API.

### [backend/07_streamlit_app.py](/Users/berkehan/Desktop/use-case-3-/backend/07_streamlit_app.py)

Affiche l'application finale utilisée par l'utilisateur.

---

## 20. Si je dois montrer une mini démo

### Démo facile à faire

1. ouvrir l'onglet `Prix par Commune` ;
2. taper `Paris` ;
3. montrer les prix et les graphiques ;
4. ouvrir l'onglet `Analyse de Bien` ;
5. coller :

`Magnifique appartement lumineux, entièrement rénové, proche transports.`

6. cliquer sur `Analyser le bien` ;
7. expliquer que le système estime un sentiment positif ;
8. terminer avec l'onglet `Résumé de Rapport` ou `Q&A Immobilier`.

### Phrase de transition

"Cette démonstration montre que le projet ne se limite pas à un script Python : il devient un véritable assistant interactif."
