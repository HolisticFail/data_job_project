import streamlit as st
import pandas as pd
import numpy as np


title = "Conclusion"
sidebar_name = "Conclusion"


def run():

    st.title(title)

    st.markdown(
        """
    ### :blue[Conclusion Projet Fil Rouge] 

    A travers les différentes étapes du projet, nous avons pu mettre à l’épreuve notre première impression du jeu de données : 

    A savoir que nous disposions de suffisamment de données pour être en mesure de dresser un profil clair et précis pour chacun de nos métiers cibles. 
    Profil qui nous aurait ainsi permis d’obtenir une classification homogène de ces métiers par nos modèles de machine learning. 

    La partie datavisualisation nous a permis de construire une cartographie des répondants et d’observer que même si les métiers de Data Analyst, Data Scientist, 
    Data Engineer et Machine Learning Engineer sont différents, ils partagent tous un tronc commun de compétences et d’utilisation de logiciels. 

    Il nous est apparu clair assez rapidement, qu’une classification des 4 métiers ne serait probablement pas possible en utilisant uniquement quelques critères de différenciation. 
    Hypothèse que nous avons pu confirmer lors de la première itération de machine learning. 

    La seconde itération quant à elle nous a amené à augmenter drastiquement le nombre de critères, en passant d’une dizaine de critères pour la première itération à deux cents pour la seconde.
    Bien qu’une augmentation du pourcentage de performance du modèle ait été observée avec l’ajout de ces nombreux critères, la classification des métiers cibles et notamment 
    des Data Engineer et Machine Learning Engineer, n’est toujours pas satisfaisante. 

    Cet état peut s’expliquer par de nombreux facteurs, que nous avons détaillé dans l’analyse de la seconde itération. L’un des principaux facteurs est très probablement, 
    le fait que les questions posées dans l’enquête ne sont pas suffisamment différenciantes pour permettre une classification précise des métiers des répondants. 

    La piste la plus pertinente à explorer pour aller plus loin dans ce travail d’analyse et de classification, serait la consolidation des données par l’ajout de nouveaux critères. 
    Que ce soit par une évolution du questionnaire du site Kaggle.com qui s’appuierait cette fois-ci directement sur des fiches de postes des différents métiers de la data pour construire son enquête.
    Ou directement par le scrapping d’offres d’emploi des métiers de la data pour être en mesure de construire le profil de ces métiers le plus précis possible.
    Il serait ainsi possible de construire un modèle qui pourrait, au travers des critères retenus proposer une classification la plus précise possible. 

    ### Appréciation du projet fil rouge 

    Ce projet fil rouge a été pour nous l’occasion d’avoir un aperçu complet du métier de Data Analyst, que ce soit pour l’étape de prise de connaissance des données, 
    la réalisation des premières analyses et conclusion. Nous avons pu également nous familiariser avec l’aspect présentation et synthèse via la création de visuels d’analyse de notre jeu de données.

        """
    )

  
