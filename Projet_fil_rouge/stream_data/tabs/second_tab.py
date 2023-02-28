import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px 


title = "Exploration des données"
sidebar_name = "Exploration des données"


def run():

    st.title(title)
    #Introduction
    st.markdown(
        """
        Le jeu de donnée à notre disposition se compose comme suit : 
        - 20036 lignes de données.  
        - 355 colonnes.

        La majorité des données sont des données catégorielles (textuelles). Il y a très peu de données numériques dans celui-ci.

        ### :blue[Cartographie des personnes qui travaillent dans la data :]

        Les modules _[matplotlib](https://matplotlib.org/)_ et _[seaborn](https://seaborn.pydata.org/)_ nous ont permis de dresser le profil des personnes travaillant dans la data à l'aide de plusieurs graphiques. 
       
       Nous avons concentré notre analyse sur les critères suivants : 

        - Origines (pays) des répondants.  
        - Niveau d'études des répondants.
        - Langage de programmation recommandé par les répondants.  
        - Salaire des répondants.
      

       Nous avons également défini des filtres pour affiner notre analyse :

        - Genre des répondants.  
        - Pays.
       """
    )
    #importation de notre dataframe

    df= pd.read_csv("data/data_dataviz.csv", sep = ";")



    #Analyse 
    st.markdown(
        """
        Afin de dresser un aperçu du profil des répondants à l'enquête Kaggle.com, nous avons décidé de nous limiter à quelques colonnes du jeu de données :
        - age  
        - gender  
        - country  
        - diploma  
        - position  
        - years_of_programming  
        - recommend_programming_to_learn_first  
        - company_size  
        - yearly_salary

        """
    ) 
    st.dataframe(df)

    #Création des graphiques

    #Palette de couleur et autres personnalisations
    custom_layer = sns.color_palette('magma_r', 12)

    custom_plot = sns.color_palette('Paired',12)

    # df_country ne contient que la colonne pays (pour geopandas)
    #Maj nom des pays
    updated_country = ({'country':{'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',
                    'Iran, Islamic Republic of...':'Iran'}})

    df = df.replace(updated_country)

    # Suppression des lignes avec Other 
    df.drop(df[df['country'] == 'Other'].index, inplace = True)

    #Fonction pour la création de graphiques 
    def diagramme_barre(variable, palette_couleurs, titre_graphique, titre_abscisse):
        sns.countplot(x=variable, palette=palette_couleurs)
        plt.xticks(rotation=90)
        plt.title(titre_graphique, fontweight='bold')
        plt.xlabel(titre_abscisse, weight = 'bold')
        plt.ylabel('Effectifs', weight = 'bold')
        plt.show();

    

    def diagramme_barre_ascending(variable, palette_couleurs, titre_graphique, titre_abscisse):
        sns.countplot(x=variable, palette=palette_couleurs, order=variable.value_counts(ascending=True).index)
        plt.xticks(rotation=90)
        plt.title(titre_graphique, fontweight='bold')
        plt.xlabel(titre_abscisse, weight = 'bold')
        plt.ylabel('Effectifs', weight = 'bold')
        plt.show();

    def diagramme_barre_descending(variable, palette_couleurs, titre_graphique, titre_abscisse):
        sns.countplot(x=variable, palette=palette_couleurs, order=variable.value_counts(ascending=False).index)
        plt.xticks(rotation=90)
        plt.title(titre_graphique, fontweight='bold')
        plt.xlabel(titre_abscisse, weight = 'bold')
        plt.ylabel('Effectifs', weight = 'bold')
        plt.show();
        
    st.markdown(
        """
        ## A quoi ressemble une personne qui travaille dans la data ?

        Bien que le terme [Data Science](https://en.wikipedia.org/wiki/Data_science) commence à apparaitre vers le milieu des années 1970, il ne désignait pas encore la science de l'analyse des données,
          mais il se référait au développement informatique.
        C'est au début des années 2000 que le terme Data Science, prend la signification qu'on lui connaît aujourd'hui. C'est avec cette observation, 
        que nous avons décidé de nous intéresser à l'âge des répondants du sondage Kaggle.com, afin de constater ou non si les personnes qui travaillent dans 
        la Data sont plutôt des personnes en tout début de carrière ou déjà en milieu fin de carrière professionnelle.
             
       """
    )

    st.markdown("### :blue[Lieu de résidence des répondants]")
    #Tri par ordre croissant
    df = df.sort_values('country')

    #Création d'un graphique qui donne une représensation des âges des répondants au sondage
    country_plot = diagramme_barre(df.country, custom_layer, "Répartition des répondants selon leur pays d'origine", 'Âge')
    st.pyplot(country_plot)
    st.markdown(
         """
         Nous constatons que les répondants sont originaires de nombreux pays différents. Une grande majorité d'entre eux vivent en Inde.
         """
    )
    #Tri par ordre croissant
    df = df.sort_values('age')

    #Création d'un graphique qui donne une représensation des âges des répondants au sondage
    age_plot = diagramme_barre(df.age, custom_layer, 'Répartition des répondants selon leur âge', 'Âge')
    #st.area_chart(chart_data)
    st.markdown('### :blue[Age des répondants :]')
    st.pyplot(age_plot)

    st.markdown(
    """Nous pouvons constater qu'une majorité de répondants au sondage a entre 18 et 35 ans, ce qui les places dans la tranche début de carrière et début de milieu de carrière professionnelle."""
        )
    

    st.markdown(
        """
        ### :blue[Diplômes des répondants :]
        Nous nous sommes ensuite intéressés au niveau d'étude des répondants, partant du postulat suivant :
        - Bachelor (bac+3) = Data Analyst.  
        - Master (bac+5) = Data Engineer / MLObs.  
        - Doctorat (bac+7) = Data Scientist. 
        """
    )


    
    #Création d'un graphique qui donne une représensation des diplômes des répondants au sondage
    diploma_dic = {'diploma' : {'Master’s degree' : 'Master', 'Bachelor’s degree' : 'Bachelor', 
                            'Doctoral degree' : 'Doctorate',
                            'Some college/university study without earning a bachelor’s degree':'no_superior_degree',
                            'Professional degree' : 'professional_degree', 
                            'I prefer not to answer' : 'no_information',
                            'No formal education past high school':'no_superior_degree'}}
    df.replace(diploma_dic, inplace = True)
    diploma_plot = diagramme_barre_descending(df.diploma, custom_layer, 'Répartition des répondants selon leur diplômes', 'Diplômes')
    st.pyplot(diploma_plot)
    
    st.markdown(
        """
        Cette première illustration nous permet de confirmer une partie de notre analyse initiale, à savoir que les personnes qui travaillent dans 
        la data ont majoritairement soit un Bachelor, Master ou Doctorat. On constate néanmoins qu'une partie des répondants n'ont pas de diplôme (du tout ou universitaire).

        Mais qu'en est-il vraiment de la répartition des diplômes en fonction de nos métiers cibles : Data Analyst, Data Engineer, MLObs et Data Scientist.
        """
    )


    # Répartition des diplômes en fonction des 3 métiers cible
    '''==> Focus sur 3 métiers de la data : Data Analyst, Data Engineer, Data Scientist :'''

    #Crréation d'une fonction Répartition des diplômes en fonction des 4 métiers cible
    def job_diploma_plot(dataframe):
            top_diploma = df[df['diploma'].isin(["Bachelor","Doctorate", "Master"])]
            top_position = top_diploma[top_diploma['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist', 'Machine Learning Engineer'])]
            table = top_position.groupby(['diploma', 'position']).size().reset_index().pivot(columns='diploma', index='position', values=0)
            plt.style.use('ggplot')

            ax=table.plot(stacked=True,kind='barh',figsize=(16,12),alpha=0.9)

            index_list = table.index.values
            total = table.values.sum()

            # Boucle de création des annotations
            for i in table.index :
                tot_x = 0
                for j in table.columns:

            # Création des légendes
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 26})
                    plt.xlabel('Effectifs',fontsize=40, weight = 'bold')
                    plt.ylabel('Emplois',fontsize=40, weight = 'bold')
                    plt.xticks(fontsize=26)
                    plt.yticks(fontsize=26)
                    plt.title('Répartition des diplômes en fonction des 4 métiers cible',fontsize=40,weight='bold');

    plotjob_diploma = job_diploma_plot(df)
    st.pyplot(plotjob_diploma)
    
    st.markdown(
         """
         Ce second graphique qui se concentre sur la répartition des diplômes en fonction de nos 4 métiers cibles, met en évidence que peu importe le métier, le diplôme le plus représenté est le Master (bac+5).
        
         """
    )
    st.markdown(
        """
        ### :blue[Langage de programmation les plus utilisés]

        Nous nous sommes ensuite intéressés aux trois langages de programmations les plus recommandés par les personnes qui travaillent dans la data.
        """
    )
    #Création d'une fonction pour créer un pie chart top 3 langages de programmation
    def progpie(dataframe):
            rec_lp = df.recommend_programming_to_learn_first.value_counts().nlargest(3).sort_values(ascending = False)
            #display(rec_lp.head())


            plt.figure(figsize=(4,4))
            plt.rcParams['font.size'] = 12.0
            #plt.rcParams["font.weight"] = "bold"
            expl = [0.1,0.1,0.2]
            plt.pie(x=rec_lp, labels = ['Python', 'SQL', 'R'],explode = expl,colors =['darkgoldenrod', 'chocolate', 'peru'], autopct = lambda x: str(round(x)) + '%',shadow = True)
            plt.title('Top 3 des langages de programmation recommandé', fontsize=18,weight='bold')
            plt.show();
    pieplotprog = progpie(df)
    st.pyplot(pieplotprog)

    st.markdown(
         """
         Sans grande surprise, il s'agit des langages : Python, SQL et R qui sont les plus recommandés par les personnes qui travaillent dans la data. 
         Ce sont en effet les principaux langages utilisés pour le stockage (SQL) et le traitement des données (Python et R).
         """
    )

    st.markdown(
         """
         Intéressons-nous maintenant à la répartition de ces 3 langages de programmation parmis nos 4 métiers cible.
    
         """
    )

    #Création d'une fonction pour créer le graphique de répartion des top 3 langages aveec nos 4métiers
    def progbyjob(dataframe):
            # Répartition des 3 langages de programmation en fonction des 4 métiers cible

            pl_df = df[df['recommend_programming_to_learn_first'].isin(['Python','R', 'SQL'])]
            pl_df_position = pl_df[pl_df['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist', 'Machine Learning Engineer'])]
            table = pl_df_position.groupby(['recommend_programming_to_learn_first','position']).size().reset_index().pivot(columns='recommend_programming_to_learn_first', index='position', values=0)
            plt.style.use('ggplot')
                
            ax=table.plot(stacked=True,kind='barh',figsize=(12,10),alpha=0.7)

            index_list = table.index.values
            total = table.values.sum()

            #ploting the annotation text
            for i in table.index :
                tot_x = 0
                for j in table.columns:
                    
                    ratio = (table.loc[(i)][j])/ total
                    x_pos = table.loc[(i)][j]+ tot_x
                    tot_x += table.loc[(i)][j]
                    if(ratio >= 0.001):
                        plt.text(x = x_pos - table.loc[(i)][j]/2, y = np.where(index_list == i)[0][0]
                                ,s= '%.1f'%(ratio*100)+'%' ,va='center', ha='center', size=10)

            #Decorating the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 14})
            plt.xlabel('Effectifs',fontsize=16, weight = 'bold')
            plt.ylabel('Emplois',fontsize=16, weight = 'bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Répartition du top 3 langage de programmation en fonction des 4 métiers cibles',fontsize=15,weight='bold');
    plotprogbyjob = progbyjob(df)
    st.pyplot(plotprogbyjob)

    st.markdown(
         """
         Nous constatons qu'au sein de nos 4 métiers cible, c'est le langage de programmation Python qui est le plus représenté. 
         """
    )

    st.markdown(
         """
         #### :blue[Salaire des répondants]

         Nous avons ensuite analysé le niveau de rémunération des répondants.
         La colonne 'yearly_salary' est composé de 725 valeurs manquantes que nous avons décidé de remplacer par la valeur médiane de la colonne.
         """
    )
    #Création d'un dictionnaire pour les valeurs de la colonne salaire
    salary_dic = {'yearly_salary':{
                        '$0-999':'500',
                        '1,000-1,999':'1500',
                        '2,000-2,999':'2500',
                        '3,000-3,999':'3500',
                        '4,000-4,999':'4500',
                        '5,000-7,499':'6500',
                        '7,500-9,999':'8500',
                        '10,000-14,999':'12500',
                        '15,000-19,999':'17500',
                        '20,000-24,999':'22500',
                        '25,000-29,999':'27500',
                        '30,000-39,999':'35000',
                        '40,000-49,999':'45000',
                        '50,000-59,999':'55000',
                        '60,000-69,999':'65000',
                        '70,000-79,999':'75000',
                        '80,000-89,999':'85000',
                        '90,000-99,999':'95000',
                        '100,000-124,999':'112500',
                        '125,000-149,999':'137500',                
                        '150,000-199,999':'175000',
                        '200,000-249,999':'225000',
                        '250,000-299,999':'275000',
                        '300,000-500,000':'400000',
                        '> $500,000':'5000000'}}
    
    #Mise en forme de la colonne salaire
    df.replace(salary_dic, inplace=True)

    #Gestion des na
    #Mise à jour des valeurs manquantes : 
    df.yearly_salary.fillna(df.yearly_salary.median(), inplace = True)
    #df.dropna(axis=0, inplace=True)
    #transformation de la variable en float64 :
    df.yearly_salary = df.yearly_salary.astype(float)
    #Vérification
    #st.dataframe(df)


    #Création d'une fonction pour créer le graphique de répartition des salaires des répondants
    def salary(dataframe):
            # Répartition des salaires des répondants

            pl_df_position = df[df['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist', 'Machine Learning Engineer'])]
            table = pl_df_position.groupby(['yearly_salary','position']).size().reset_index().pivot(columns='yearly_salary', index='position', values=0)
            plt.style.use('ggplot')
                
            ax=table.plot(stacked=True,kind='barh',figsize=(12,10),alpha=0.7)

            index_list = table.index.values
            total = table.values.sum()

            #ploting the annotation text
            for i in table.index :
                tot_x = 0
                for j in table.columns:
                    
                    ratio = (table.loc[(i)][j])/ total
                    x_pos = table.loc[(i)][j]+ tot_x
                    tot_x += table.loc[(i)][j]
                    if(ratio >= 0.001):
                        plt.text(x = x_pos - table.loc[(i)][j]/2, y = np.where(index_list == i)[0][0]
                                ,s= '%.1f'%(ratio*100)+'%' ,va='center', ha='center', size=10)

            #Decorating the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 14})
            plt.xlabel('Effectifs',fontsize=16, weight = 'bold')
            plt.ylabel('Emplois',fontsize=16, weight = 'bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Répartition des salaires en fonction des 4 métiers cibles',fontsize=15,weight='bold');
    plotsalry = salary(df)
    st.pyplot(plotsalry)

    st.markdown(
         """
         Nous pouvons observer une grande disparité dans les salaires quelque soit le métier cible. 
         Cela s'explique par le fait que les répondants au sondage Kaggle.com ne sont pas tous issue du même pays. 
         Comme nous avions pu le constater dans le graphique du pays d'origine des répondants, un grand nombre d'entre eux vivent en Inde. 
         Dans les prochains graphiques, nous nous intéresserons à la France pour retirer ses disparités.
         """
    )

    st.markdown(
         """
         ### :blue[Répartitions des métiers ciblesen France]
         """
    )
    def jobinfr(dataframe):
# Répartition des 3 métiers cible en France

            only_france = df[df['country'].isin(['France'])]
            top_position_france = only_france[only_france['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist','Machine Learning Engineer'])]
            table = top_position_france.groupby(['position', 'country']).size().reset_index().pivot(columns='position', index='country', values=0)
            plt.style.use('ggplot')
                
            ax=table.plot(stacked=True,kind='barh',figsize=(12,10),alpha=0.7)

            index_list = table.index.values
            total = table.values.sum()

            #ploting the annotation text
            for i in table.index :
                tot_x = 0
                for j in table.columns:
                    
                    ratio = (table.loc[(i)][j])/ total
                    x_pos = table.loc[(i)][j]+ tot_x
                    tot_x += table.loc[(i)][j]
                    if(ratio >= 0.001):
                        plt.text(x = x_pos - table.loc[(i)][j]/2, y = np.where(index_list == i)[0][0]
                                ,s= '%.1f'%(ratio*100)+'%' ,va='center', ha='center', size=10)

            #Decorating the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 14})
            plt.xlabel('Effectifs',fontsize=16, weight = 'bold')
            plt.ylabel('Pays',fontsize=16, weight = 'bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Répartition des 4 métiers cibles en France',fontsize=15,weight='bold');
    jobfr = jobinfr(df)
    st.pyplot(jobfr)

    st.markdown(
         """
         Nous pouvons constater que les répondants français sont en majorité des Data Scientist(70%) suivi d'assez loin par les Machine Learning Engineer (16) et les Data Analyst(11% des répondants)
         """
    )
    st.markdown(
         """
         ### :blue[Salaire des répondants français]
         """
    )
    def salfr(dataframe):
# Répartition des 3 métiers cible en France

            only_france = df[df['country'].isin(['France'])]
            top_position_france = only_france[only_france['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist','Machine Learning Engineer'])]
            table = top_position_france.groupby(['position', 'yearly_salary']).size().reset_index().pivot(columns='position', index='yearly_salary', values=0)
            plt.style.use('ggplot')
                
            ax=table.plot(stacked=True,kind='barh',figsize=(12,10),alpha=0.7)

            index_list = table.index.values
            total = table.values.sum()

            #ploting the annotation text
            for i in table.index :
                tot_x = 0
                for j in table.columns:
                    
                    ratio = (table.loc[(i)][j])/ total
                    x_pos = table.loc[(i)][j]+ tot_x
                    tot_x += table.loc[(i)][j]
                    if(ratio >= 0.001):
                        plt.text(x = x_pos - table.loc[(i)][j]/2, y = np.where(index_list == i)[0][0]
                                ,s= '%.1f'%(ratio*100)+'%' ,va='center', ha='center', size=10)

            #Decorating the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 14})
            plt.xlabel('Effectifs',fontsize=16, weight = 'bold')
            plt.ylabel('Pays',fontsize=16, weight = 'bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Répartition des 4 métiers cibles en France',fontsize=15,weight='bold');
    salary_france = salfr(df)
    st.pyplot(salary_france)

    st.markdown(
         """
         Nous pouvons constater que la plage des salaires commence à 500€ et culmine à 400k€. Une grande majorité des répondants est concentrée entre 17k€ et 65k€.
         Nous observons néanmoins que les salaires les plus hauts ne semblent concerner qu'exclusivement les Data Scientists. Comme nous avions pu l'observer sur le graphique précédent, 
         ceux-ci sont surreprésentés en France. 
         """
    )

    st.markdown(
         """
         ### :blue[Répartitions des 4 métiers cibles en fonction du genre en France]
         """
    )

    def genrefr(dataframe):
         # Répartition des 3 métiers cible en fonction du genre 

            gender_df = df[df['gender'].isin(['Man','Woman'])]
            gender_df_france = gender_df[gender_df['country'].isin(['France'])]
            gender_df_position = gender_df_france[gender_df_france['position'].isin(['Data Analyst','Data Engineer', 'Data Scientist','Machine Learning Engineer'])]
            table = gender_df_position.groupby(['position', 'gender']).size().reset_index().pivot(columns='gender', index='position', values=0)
            plt.style.use('ggplot')
                
            ax=table.plot(stacked=True,kind='barh',figsize=(12,10),alpha=0.7)

            index_list = table.index.values
            total = table.values.sum()

            #ploting the annotation text
            for i in table.index :
                tot_x = 0
                for j in table.columns:
                    
                    ratio = (table.loc[(i)][j])/ total
                    x_pos = table.loc[(i)][j]+ tot_x
                    tot_x += table.loc[(i)][j]
                    if(ratio >= 0.001):
                        plt.text(x = x_pos - table.loc[(i)][j]/2, y = np.where(index_list == i)[0][0]
                                ,s= '%.1f'%(ratio*100)+'%' ,va='center', ha='center', size=10)

            #Decorating the plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size': 14})
            plt.xlabel('Effectifs',fontsize=16, weight = 'bold')
            plt.ylabel('Emplois',fontsize=16, weight = 'bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('Répartition des 4 métiers cibles en fonction du genre en France',fontsize=15,weight='bold');
    genrejobfr = genrefr(df)
    st.pyplot(genrejobfr)

    st.markdown(
         """
         Nous pouvons observer qu'en-dehors du métier de Data Analyst, nos autres métiers cible sont à dominance masculine. 
         """
    )

    st.markdown(
         """
         ## Profil de la personne qui travaillent dans la data !

         Comme analysés tout au long de cette page, nous sommes en mesure d'établir un profil type de la personne qui travaille dans le domaine de la data grâce à l'enquête réalisée par le site Kaggle.com.

         Il s'agit donc dans une majorité des cas d'un homme qui a entre 18 et 35 ans et qui possède un Master ou un Bachelor et qui a pour habitude d'utiliser 
         les langages de programmation suivant : Python, SQL et R. 

         ### :blue[Conclusion]

         Le portrait que nous avons dressé ce base sur quelques critères du jeu de données qui était à notre disposition pour ce projet fil rouge. Et est un profil simplifié de ce qui est attendu pour 
         les gens qui travaillent dans la data. Néanmoins, ce profil ne permet pas d'identifier rapidement si la personne est un Data Analyst, un Data Scientist, un Data Engineer ou un MLObs.
         Dans la page suivante, nous allons essayer au travers du machine learning de réaliser un algorithme de classification et ainsi vérifier si 
         le jeu de données dispose de suffisamment d'éléments différentiant pour classer nos 4 métiers cible.
         """
    )