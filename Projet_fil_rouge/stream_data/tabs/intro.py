import streamlit as st
from PIL import Image


title = "Data Job - Profil des personnes qui évoluent dans le milieu de la data"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    st.image(Image.open("assets/data.png"))

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        **:blue[Sujet :]** [Analyse des techniques et outils utilisés par les professionnels de la Data](https://www.kaggle.com/c/kaggle-survey-2020/overview)

**:blue[Nature des données :]** Notre projet s’appuie sur une enquête réalisée par le site Kaggle.com. Elle est composée de 39 questions, 18 d’entre elles sont des questions à choix unique, le reste étant à choix multiple. 

L’enquête a été réalisée sur une période de 3 mois, de juillet 2020 à septembre 2020 et a été transmise à l’ensemble des personnes qui sont inscrites sur le site Kaggle.com. 

Un total de 171 nationalités ont pris part à cette enquête. 

Initialement, ce jeu de données a été constitué pour la compétition annuelle Machine Learning et Data Science de Kaggle qui consiste à réaliser un notebook qui présente et raconte l’histoire des données récoltées dans l’enquête.

**:blue[Quel est l’objectif du projet ?]**

●	Identifier les différents profils techniques qui se sont créés dans l’industrie de la Data.  
●	Quelles sont les tâches effectuées par les personnes qui travaillent dans la Data ?  
●	Quels sont les outils qui sont utilisés par les personnes qui travaillent dans la Data ?  
●	Définir quels sont les outils et les compétences attendues par les personnes qui travaillent / souhaitent travailler dans la data.

        """
    )
