import streamlit as st
from PIL import Image


title = "Machine Learning : Classification de nos 4 métiers cible"
sidebar_name = "Machine Learning : Classification de nos 4 métiers cible"


def run():

    st.title(title)

    st.markdown(
        """
        ### :blue[Méthodologie] :

        L’objectif de notre projet fil rouge est de réussir à identifier les différents profils des personnes qui travaillent dans le milieu de la data, soit à réaliser une classification de ceux-ci. 
        Nous avons fait le choix de mettre à l'épreuve les deux modèles suivants : Decision Tree et Random Forest.   
        
        Nous avons réalisé deux itérations sur ces deux modèles, ci-après les résultats que nous avons obtenus :

        ### :blue[Première itération de machine learning]

        #### Périmètre d'analyse : 

        Lors de la première itération, nous nous sommes basés sur un jeu de données réduit. Nous avons sélectionné 18 critères et avons réalisé un filtre sur la plage de données 'Job' afin de ne sélectionner que nos 4 métiers cible (Data Analyst, Data Scientist, Data Engineer et MLObs).
        Nous avons appliqué la stratégie suivante pour le préprocessing : 
        - Traitement des valeurs numériques manquantes : remplacer par la valeur médiane.
        - Traitement des valeurs catégorielles manquantes : remplacer par le mode. 
        - Simplification de certaines valeurs (exemple : Bachelor's degree = bachelor ; A personal computer or laptop = personal_computer, etc.).

        Nous avons ensuite appliqué une standardisation des valeurs numériques et avons utilisé la méthode get_dummies pour les valeurs catégorielles.

        Et enfin, nous avons utilisé la méthode RandomOverSampler pour homogénéiser nos 4 classes / métiers cible.

        Ci-après nos résultats :
        """
        )
    st.markdown(
        """
        _Decision Tree_ : 
        """
    )
    st.image(Image.open("data/it1_dt.png"))
    st.image(Image.open("data/hm_dt_it1.png"))

    st.markdown(
        """
        _Random Forest_ : 
        """
    )
    st.image(Image.open("data/it1_rf.png"))
    st.image(Image.open("data/hm_rf_it1.png"))
    
    st.markdown(
        """
        ### Conclusion itération 1 : 

        Bien que les scores de précisions soient relativement élevés pour chaque modèle, nous pouvons constater au travers des matrices de confusion que certains de nos métiers cible ne sont pas correctement classés.
        C'est fort de ce constat que nous avons décidé d'élargir et de revoir les critères sélectionnés pour la seconde itération. Nous nous sommes dès lors fixés comme objectif d'améliorer la classification des métiers : Data Engineer et MLObs.
        """
    )
    st.markdown(
        """
        ### :blue[Seconde itération]

        #### Périmètre d'analyse :

        Pour cette seconde itération, nous avons consulté les fiches métiers des Data Engineer et MLObs pour nous aider à sélectionner les colonnes du jeu de données qui correspondaient le plus à ces deux métiers.
        Nous avons également fait le choix de ne sélectionner que des données catégorielles et avons élargis la plage de critères de 17 à 199.

        De plus, nous avons appliqué une stratégie sensiblement différente pour le préprocessing : 
        - Traitement des valeurs catégorielles manquantes : remplacer par 'no_reply'.

        La méthode get_dummies et le RandomOverSampler ont également été utilisé pour cette seconde itération.

        Ci-après nos résultats : 
        """
    )
    st.markdown(
        """
        _Decision Tree_ : 
        """
    )
    st.image(Image.open("data/it2_dt.png"))
    st.image(Image.open("data/hm_dt_it2.png"))

    st.markdown(
        """
        _Random Forest_ : 
        """
    )
    st.image(Image.open("data/it2_rf.png"))
    st.image(Image.open("data/hm_rf_it2.png"))
    
    st.markdown(
        """
        ### Conclusion itération 2 : 

        Nous constatons que l'ajout de nouveaux critères a permis aux modèles de gagner en précision (env. +5% vs la première itération).
        Néanmoins, nous ne constatons qu'une faible progression dans la classification des métiers : Data Engineer et MLObs.
        Ci-après nous avons essayé de trouver des explications à ce problème de classification :

        - La raison la plus probable quant à ce problème de classification, vient de la nature de nos données. Pour rappel, il s'agit des réponses à une enquête 
        réalisée par le site Kaggle sur les compétences et logiciels utilisés par les personnes qui travaillent dans la data. 
        Il est possible que les questions posées soient trop générales et ne permettent pas de différencier suffisamment les répondants.

        - Il est également possible que les personnes qui ont participé à l'enquête occupent des postes sur lesquels ils réalisent aussi bien des tâches propres aux : 
        Data Analyst, Data Scientist, Data Engineer et Machine Learning Engineer.

        - Une autre possibilité est également un enthousiasme trop prononcé dans les réponses aux questionnaires, c'est-à-dire indiquer l'utilisation de 
        tels ou tels logiciels ou compétences, car ils en ont connaissance et/ou l'ont déjà utilisé, sans que cela ne reflète strictement leurs activités journalières.

        - Enfin, il est bon de rappeler que Kaggle est une plateforme d'apprentissage et que les personnes qui visitent ce site le font avant tout pour se former. De ce fait, 
        ce n'est pas surprenant d'avoir des Data Analyst, Data Scientist, Data Engineer et Machine Learning Engineer qui partagent des compétences et connaissances similaires.  
        
        ### :blue[Conclusion]
        Ces hypothèses nous amènent à penser que nous avons atteint un niveau de performance satisfaisant avec nos modèles de machine learning au regard des données à notre disposition. 
        Pour améliorer encore plus la classification, il faudrait envisager l’utilisation d’un nouveau jeu de données, qui cette fois-ci, s'appuierait par exemple sur des données provenant 
        directement d'offres d'emploi propres à chacun des métiers cibles que nous avons étudié.
        
        """
    )