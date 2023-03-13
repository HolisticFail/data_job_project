import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import f1_score 
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import chime 

title = "Modèle Machine Learning"
sidebar_name = "Modèle Machine Learning"


def run():
            
    #liste dataviz
    matplotlib_list = [1,0]
    seaborn_list = [1,0]
    plotly_list = [1,0]
    geoplotlib_list = [1,0]
    none_viz_list = [1,0]
    
    
    #liste machine learning skill
    scikitlearn_list = [1,0]
    tensorflow_list = [1,0]
    keras_list = [1,0]
    pytorch_list = [1,0]
    fastai_list = [1,0]
    none_ml_list = [1,0]
       
    #liste machine learning model
    linear_list = [1,0]
    decision_model_list = [1,0]
    gradient_model_list = [1,0]
    dense_model_list = [1,0]
    convolutional_model_list = [1,0]
    none_model_list = [1,0]
    
    #liste main activity
    act_analyse_list = [1,0]
    act_build_list = [1,0]
    act_model_activity_list = [1,0]
    act_model_up_list = [1,0]

    st.markdown(
          """
          ## Modèle de machine learning
        Nous avons entraîné un modèle réduit avec 21 critères afin de faire une démonstration du fonctionnement de notre algorithme de classification de nos 4 métiers cible. 
        Afin de simplifier la démonstration nous avons sélectionné quelques compétences de programmations en dataviz, en machine learning et les tâches principales qu'exécutent le répondant.
        Notre modèle Random Forest d'entraînement obtient un score de précision de 69,56%. Et permet une classification correcte des différents métiers cible (avec là encore des performances assez inégales pour les métiers de : Data Engineer et de MLObs).
          ### :blue[A vous de jouer !]
          Nous vous invitons à sélectionner entre 1 (oui) ou 0 (non) pour chaque champ afin de voir quel est votre métier de rêve dans la data !
          """
    )
    
    def class_model():

            matplotlib_skill = st.selectbox('Utilisez-vous matplotlib ?', matplotlib_list)
            seaborn_skill = st.selectbox('Utilisez-vous seaborn ?', seaborn_list)
            plotly_skill = st.selectbox('Utilisez-vous plotly ?', plotly_list)
            geoplotlib_skill = st.selectbox('Utilisez-vous geoplotlib ?', geoplotlib_list)
            none_viz_skill = st.selectbox("Vous n'utilisez pas de librairie de dataviz", none_viz_list)
            scikitlearn_skill = st.selectbox('Utilisez-vous scikit-learn ?', scikitlearn_list)
            tensorflow_skill = st.selectbox('Utilisez-vous tensorflow ?', tensorflow_list)
            keras_skill = st.selectbox('Utilisez-vous keras ?', keras_list)
            pytorch_skill = st.selectbox('Utilisez-vous pytorch ?', pytorch_list)
            fastai_skill = st.selectbox('Utilisez-vous fastai ?', fastai_list)
            none_ml_skill = st.selectbox("Vous n'utilisez pas de librairie de machine learning", none_ml_list)
            linear_skill = st.selectbox('Utilisez-vous les modèles de regression linéaire ?', linear_list)
            decision_skill = st.selectbox("Utilisez-vous les modèles d'arbre de décision ?", decision_model_list)
            gradient_skill = st.selectbox('Utilisez-vous les modèles de gradient boosting ?', gradient_model_list)
            dense_skill = st.selectbox('Utilisez-vous les modèles dense neural network ?', dense_model_list)
            convolutional_skill = st.selectbox('Utilisez-vous les modèles convolutional neural networks ?', convolutional_model_list)
            none_model_skill = st.selectbox("Vous n'utilisez pas de modèle de machine learning", none_model_list)
            act_analyse__skill = st.selectbox("Votre activité principale consiste en l'analyse des données", act_analyse_list)
            act_build_skill = st.selectbox('Votre activité principale consiste en la création de base de donnée', act_build_list)
            act_model_activity_skill = st.selectbox('Votre activité principale consiste en la création de modèle de machine learning', act_model_activity_list)
            act_model_up_skill = st.selectbox("Votre activité principale consiste en l'amélioration des modèles de machine learning", act_model_up_list)
    
            
            data = {
                  'matplotlib_skill':matplotlib_skill,
                  'seaborn_skill':seaborn_skill,
                  'plotly_skill':plotly_skill,
                  'geoplotlib_skill':geoplotlib_skill,
                  'none_viz_skill':none_viz_skill,
                  'scikitlearn_skill':scikitlearn_skill,
                  'tensorflow_skill':tensorflow_skill,
                  'keras_skill':keras_skill,
                  'pytorch_skill':pytorch_skill,
                  'fastai_skill':fastai_skill,
                  'none_ml_skill':none_ml_skill,
                  'linear_skill':linear_skill,
                  'decision_skill':decision_skill,
                  'gradient_skill':gradient_skill,
                  'dense_skill':dense_skill,
                  'convolutional_skill':convolutional_skill,
                  'none_model_skill':none_model_skill,
                  'act_analyse__skill':act_analyse__skill,
                  'act_build_skill':act_build_skill,
                  'act_model_activity_skill':act_model_activity_skill,
                  'act_model_up_skill':act_model_up_skill,
            }
            param_classification = pd.DataFrame(data, index = [0])
            return param_classification
    df1 = class_model()
    
    st.write(df1)
    features = pd.get_dummies(df1, dtype = np.int64)
    pickled_model = pkl.load(open('data/rf1.pkl', 'rb'))
    prediction = pickled_model.predict(features)

    st.write('Votre métier de rêve dans la data est :',prediction)