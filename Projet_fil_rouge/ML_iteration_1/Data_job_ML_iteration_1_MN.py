#!/usr/bin/env python
# coding: utf-8

# # Preprocessing DATA Job

# Auteurs : Monya M. - Nejib B.

# ## Rappel :
# 
# Notre jeu de données provient d'une enquête réalisée par le site Kaggle. Il s'agit d'un questionnaire à choix multiple de 36 questions. Toutes les questions ne sont pas à choix multiples.
# 
# Nous avons fait le choix pour ce premier tour de machine learning de nous concentrer sur les 18 questions uniques de l'enquête. Nous avons de plus effectué un premier tri depuis Excel sur la colonne 'Position' qui référence les emplois des répondants, pour nous concentrer sur nos **4 métiers cibles** : *Data Scientist*, *Data Analyst*, *Data Engineer* et *Machine Learning Engineer*.
# 
# Dans de prochaines itérations nous étendrons le périmètre d'analyse et de preprocessing.
# 
# 
# ## Périmètre :
# 
# Pour cette première étape de preprocessing nous avons fait le choix de nous concentrer sur les **18 colonnes uniques** de notre jeu de données et pour un total de **5670 lignes**. 

# # PREPROCESSING - Etape 1 :

# ## Import du jeu de données :

# In[1]:


import pandas as pd
import numpy as np

#importation du jeu de données
df = pd.read_csv('ml_data_2.csv', sep = ';', index_col = 0)

#Affichage des 5 premières lignes du jeu de données et de sa taille
display(df.head())
df.shape


# ### Affichage du jeu de données :
# 
# Nous ne constatons aucun problème lors de l'importation et de l'affichage de notre jeu de données. Nous disposons bien également de nos 18 colonnes et de nos 5670 lignes.

# ## Informations et détails du jeu de données : 

# In[2]:


#Afficher les informations de df :
display(df.info())

#Lister les colonnes de df : 
display(df.columns)

#Afficher la répartition des valeurs manquantes dans df : 
display(df.isna().sum())


# ## Informations essentielles : 

# In[3]:


#Fonction qui permet l'affichage des informations suivantes : type, % des valeurs manquantes, nbre de valeurs uniques, valeur unique, valeur moyenne ou valeur la plus fréquente
def summary(df):
    table = pd.DataFrame(
        index=df.columns,
        columns=['type_info', '%_missing_values', 'nb_unique_values'])
    table.loc[:, 'type_info'] = df.dtypes.values
    table.loc[:, '%_missing_values'] = (df.isna().sum().values / len(df))*100
    table.loc[:, 'nb_unique_values'] = df.nunique().values

    def get_list_unique_values(df):
        dict_ = {}
        for col in df.columns:
            if df[col].nunique() < 6:
                dict_[col] = list(df[col].unique())
            else:
                if df[col].dtypes == "O":
                    dict_[col] = "Too much categories..."
                else:
                    dict_[col] = "Too much values..."
        return pd.DataFrame({'unique_values': dict_.values()},
                            index=df.columns)

    infos_table_int1 = pd.merge(table,
                                get_list_unique_values(df),
                                left_index=True,
                                right_index=True)

    def get_mean_mode(df):
        dict_ = {}
        for col in df.columns:
            if df[col].dtypes == "O":
                dict_[col] = df[col].mode()[0]
            else:
                dict_[col] = df[col].mean()
        return pd.DataFrame(pd.Series(dict_), columns=["mean_mode_values"])

    infos_table_int2 = pd.merge(infos_table_int1,
                                get_mean_mode(df),
                                left_index=True,
                                right_index=True)

    def alerts(df):
        thresh_na = 0.25
        thresh_balance = 0.8
        dict_ = {}
        for col in df.columns:
            if (df[col].count() / len(df)) < thresh_na:
                dict_[col] = "Too much missing values ! "
            elif df[col].value_counts(
                    normalize=True).values[0] > thresh_balance:
                dict_[col] = "It's imbalanced !"
            else:
                dict_[col] = "Nothing to report"

        return pd.DataFrame(pd.Series(dict_), columns=["flag"])

    infos_table = pd.merge(infos_table_int2,
                           alerts(df),
                           left_index=True,
                           right_index=True)

    return infos_table

#Affichage des informations :
summary(df)


# A l'aide de ce tableau et de la colonne **'mean_mode_values'**, nous serons en mesure de prendre les meilleurs décisions dans la gestion des valeurs manquantes.

# #### Premiers constats :  

# **1/** Nous remarquons via la métode **'info( )'** que toutes nos variables ont pour type **'object'**, cela provient du fait que même pour les colonnes qui contiennent des nombres, il y a également du texte. Un premier travail de retraitement des données sera nécessaire.
# 
# **2/** Nous remarquons également qu'un grand nombre de colonnes contiennent des valeurs manquantes. Les deux colonnes les plus problématiques sont : **bigdata_product_used** et **bi_tool_used**, il sera donc préférable de supprimer ces deux colonnes de notre dataframe lors du préprocessing. 
# 
# **3/** Les autres colonnes qui contiennent des valeurs manquantes seront gérées au cas par cas. Soit à l'aide de la colonne **'mean_mode_values'**, soit par la **suppression** des valeurs manquantes.

# # PREPROCESSING - Etape 2 : Analyse et retraitement des colonnes : 

# Ci-après nous allons détailler les modifications que nous allons apporter à chaque colonne de notre dataframe.

# ## Colonne age : non pertinent pour notre modèle de machine learning, à supprimer

# ## Colonne genre : non pertinent pour notre modèle de machine learning, à supprimer

# ## Colonne pays : non pertinent pour notre modèle de machine learning, à supprimer

# Nous pouvons d'ors et déjà supprimer **les 3 premières colonnes** de notre dataframe, car elles **n'apportent pas d'information utile** dans l'identification du métier d'un répondant. 

# In[5]:


#Création d'un nouveau dataframe dans lequel nous allons réaliser les modifications importantes à notre jeu de données :
df_prep = df.drop(['age', 'gender', 'country'], axis = 1).copy()


#Vérification que les modifications ont bien été apportées à df_prep
df_prep.head()


# ## Colonne diplome : pertinent pour notre modèle de machine learning => feature
# 

# In[6]:


#Affichage des valeurs uniques et manquantes : 
print('Les valeurs uniques sont :',df_prep.diploma.unique())
print('Il y a :',df_prep.diploma.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# A l'aide de la méthode **'unique( )'** nous récupérons l'ensemble des valeurs uniques de la colonne **'diploma'**. 

# In[7]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
diploma_dic = {'diploma' : {'Master’s degree' : 'master', 'Bachelor’s degree' : 'bachelor', 
                            'Doctoral degree' : 'doctorate',
                            'Some college/university study without earning a bachelor’s degree':'no_superior_degree',
                            'Professional degree' : 'professional_degree', 
                            'I prefer not to answer' : 'no_information',
                            'No formal education past high school':'no_superior_degree'}}


# Nous effectuons un retraitement des valeurs de la colonne **'diploma'** afin de supprimer les informations inutiles.

# In[8]:


#Mise à jour des valeurs de la colonne 'diploma'
df_prep.replace(diploma_dic, inplace = True)

#Vérification que tout s'affiche correctement.
display(df_prep.head())
print('Les nouvelles valeurs uniques sont :',df_prep.diploma.unique())


# ## Colonne job : pertinent pour notre modèle de machine learning => valeur cible

# La colonne job contient l'ensemble de nos valeurs cibles. Nous allons donc la retirer de df_prep et l'ajouter dans une nouvelle variable **'target'**.

# In[9]:


#Création d'une nouvelle variable dat : 
target = df_prep.job.copy()

#Affichage des informations : 
display(target.head())
display(target.shape)


# **Intéressons nous à la répartition de nos 4 métiers cibles :** 

# In[10]:


#Pourcentage de répartition des 4 métiers cibles
target.value_counts(normalize=True)*100


# 
# Nous constatons d'emblée une disparité entre les classes de notre variable **target**. La classe Data Engineer est sous représentée et elle risque d'être traitée comme une valeur aberrante par notre modèle de machine learning.
# 
# Il est donc primordial d'appliquer une méthode de **rééchantillonnage** soit à l'aide d'un **sur-échantillonnage** (Oversampling) ou d'un **sous-échantillonnage** (Undersampling).
# 
# Pour cette première itération de machine learning, nous ferons 2 essaies de sur-échantillonnage et 2 essaies de sous-échantillonnage, à l'aide des méthodes suivantes : 
# 
# **Oversampling :** Oversampling aléatoire (*RandomOverSampler*) et SMOTE (*SMOTE*).  
# *Formules :* RandomOverSampler().fit_resample(X_train, y_train) // SMOTE().fit_resample(X_train, y_train)
# 
# **Undersampling :** Undersampling aléatoire (*RandomUnderSampler*) et l'application de ClusterCentroids (*ClusterCentroids*).  
# *Formules :* RandomUnderSampler().fit_resample(X_train, y_train) // ClusterCentroids().fit_resample(X_train, y_train)

# ## Colonne years_writting_code : pertinent pour notre modèle de machine learning => feature

# In[11]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.years_writting_code.unique())
print('Il y a :',df_prep.years_writting_code.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# A l'aide de la méthode **'unique( )'** nous récupérons l'ensemble des valeurs uniques de la colonne **'years_writting_code'**. 

# In[12]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
year_programming_dic = {'years_writting_code' : {'5-10 years':'7.5' , '< 1 years': '1', 
                                                 '3-5 years': '4', '10-20 years':'15', 
                                                 '1-2 years' : '1','20+ years':'20', 
                                                 'I have never written code':'0',
                                                }}


# Nous effectuons un retraitement des valeurs de la colonne **'years_writting_code'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **52** valeurs manquantes (soit **1%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[13]:


#Mise à jour des valeurs de la colonne 'years_writting_code' :
df_prep.replace(year_programming_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.years_writting_code.fillna(df_prep.years_writting_code.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.years_writting_code = df_prep.years_writting_code.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.years_writting_code.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.years_writting_code.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.years_writting_code.dtype)


# ## Colonne most_recommended_langage : pertinent pour notre modèle de machine learning => feature

# In[14]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.most_recommended_langage.unique())
print('Il y a :',df_prep.most_recommended_langage.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# La colonne comprend **254** valeurs manquantes (soit **4,5%** de données manquantes), nous avons décidé de remplacer celles-ci par la valeur moyenne de la colonne. 

# In[15]:


#Mise à jour des valeurs manquantes : 
df_prep.most_recommended_langage.fillna('Python', inplace = True)

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.most_recommended_langage.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# ## Colonne plateform_used_for_datascience : pertinent pour notre modèle de machine learning => feature

# In[17]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.plateform_used_for_datascience.unique())
print('Il y a :',df_prep.plateform_used_for_datascience.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# Nous effectuons un retraitement des valeurs de la colonne **'plateform_used_for_datascience'**, nous simplifions les intitulés des valeurs.
# 
# Cette colonne comprend **391** valeurs manquantes (soit **7%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[18]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
plateform_dic = {'plateform_used_for_datascience':{'A personal computer or laptop' : 'personal_computer',
       'A cloud computing platform (AWS, Azure, GCP, hosted notebooks, etc)':'coud_computing_plateform',
       'A deep learning workstation (NVIDIA GTX, LambdaLabs, etc)': 'deep_learning_workstation', 
                                                   'Other':'other', 'None':'none'}}


# In[19]:


#Mise à jour des valeurs de la colonne 'plateform_used_for_datascience' :
df_prep.replace(plateform_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.plateform_used_for_datascience.fillna('personal_computer', inplace = True)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.plateform_used_for_datascience.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.plateform_used_for_datascience.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# ## Colonne tpu_used : pertinent pour notre modèle de machine learning => feature (à voir)

# In[20]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.tpu_used.unique())
print('Il y a :',df_prep.tpu_used.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[21]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
tpu_dic = {'tpu_used': {'2-5 times':'3.5',
                       'Never':'0',
                       '6-25 times': '15.5',
                       'Once':'1',
                       'More than 25 times':'25'}}


# Nous effectuons un retraitement des valeurs de la colonne **'tpu_used'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **423** valeurs manquantes (soit **7,5%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[23]:


#Mise à jour des valeurs de la colonne 'tpu_used' :
df_prep.replace(tpu_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.tpu_used.fillna(df_prep.tpu_used.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.tpu_used = df_prep.tpu_used.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.tpu_used.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.tpu_used.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.tpu_used.dtype)


# ## Colonne years_of_use_of_ml : pertinent pour notre modèle de machine learning => feature

# In[24]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.years_of_use_of_ml.unique())
print('Il y a :',df_prep.years_of_use_of_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[25]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
years_use_ml_dic = {'years_of_use_of_ml':{'1-2 years':'1',
                                          '3-4 years':'3.5',
                                          '2-3 years':'2.5',
                                          'Under 1 year':'1',
                                          '4-5 years':'4.5',
                                          '20 or more years':'20',
                                          '5-10 years':'7.5',
                                          '10-20 years':'15',
                                          'I do not use machine learning methods':'0'}}


# Nous effectuons un retraitement des valeurs de la colonne **'years_of_use_of_ml'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **503** valeurs manquantes (soit **9%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[26]:


#Mise à jour des valeurs de la colonne 'years_of_use_of_ml' :
df_prep.replace(years_use_ml_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.years_of_use_of_ml.fillna(df_prep.years_of_use_of_ml.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.years_of_use_of_ml = df_prep.years_of_use_of_ml.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.years_of_use_of_ml.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.years_of_use_of_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.years_of_use_of_ml.dtype)


# ## Colonne company_size : pertinent pour notre modèle de machine learning => feature 

# In[27]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.company_size.unique())
print('Il y a :',df_prep.company_size.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[28]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
company_size_dic = {'company_size':{'10,000 or more employees':'10000',
                                     '250-999 employees':'625',
                                     '1000-9,999 employees':'5000',
                                     '0-49 employees':'25',
                                     '50-249 employees':'150'}}


# Nous effectuons un retraitement des valeurs de la colonne **'company_size'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **444** valeurs manquantes (soit **8%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[29]:


#Mise à jour des valeurs de la colonne 'company_size' :
df_prep.replace(company_size_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.company_size.fillna(df_prep.company_size.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.company_size = df_prep.company_size.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.company_size.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.company_size.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.company_size.dtype)


# ## Colonne how_many_ppl_work_on_ml : non pertinent pour notre modèle à supprimer

# In[30]:


#Suppression de la colonne :
df_prep = df_prep.drop(['how_many_ppl_work_on_ml'], axis = 1)


# ## Colonne do_your_company_use_ml : pertinent pour notre modèle de machine learning => feature

# In[31]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.do_your_company_use_ml.unique())
print('Il y a :',df_prep.do_your_company_use_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[32]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
ml_use_dic = {'do_your_company_use_ml':{'We have well established ML methods (i.e., models in production for more than 2 years)':'yes',
       'We are exploring ML methods (and may one day put a model into production)':'not_yet',
       'No (we do not use ML methods)':'no', 'I do not know':'no',
       'We recently started using ML methods (i.e., models in production for less than 2 years)':'yes',
       'We use ML methods for generating insights (but do not put working models into production)':'yes'}}


# Nous effectuons un retraitement des valeurs de la colonne **'do_your_company_use_ml'**, nous simplifions les intitulés des valeurs.
# 
# Cette colonne comprend **562** valeurs manquantes (soit **10%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **moyenne** de la colonne, soit **yes**.

# In[33]:


#Mise à jour des valeurs de la colonne 'do_your_company_use_ml' :
df_prep.replace(ml_use_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.do_your_company_use_ml.fillna('yes', inplace = True)


#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.do_your_company_use_ml.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.do_your_company_use_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# ## Colonne salary : pertinent pour notre modèle de machine learning => feature

# In[34]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.salary.unique())
print('Il y a :',df_prep.salary.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[35]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
salary_dic = {'salary':{
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


# Nous effectuons un retraitement des valeurs de la colonne **'salary'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **725** valeurs manquantes (soit **12%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[36]:


#Mise à jour des valeurs de la colonne 'salary' :
df_prep.replace(salary_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.salary.fillna(df_prep.salary.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.salary = df_prep.salary.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.salary.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.salary.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.salary.dtype)


# ## Colonne money_spent_in_ml : pertinent pour notre modèle de machine learning => feature 

# In[37]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.money_spent_in_ml.unique())
print('Il y a :',df_prep.money_spent_in_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[38]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
money_spent_ml_dic = {'money_spent_in_ml':{'$100,000 or more ($USD)':'100000',
                                           '$10,000-$99,999':'50000',
                                           '$1000-$9,999':'5000',
                                           '$0 ($USD)':'0', 
                                           '$1-$99':'50', 
                                           '$100-$999':'500'}}


# Nous effectuons un retraitement des valeurs de la colonne **'money_spent_in_ml'**, nous calculons la valeur moyenne pour chacune des valeurs uniques.
# 
# Cette colonne comprend **794** valeurs manquantes (soit **14%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **médiane** de la colonne.

# In[39]:


#Mise à jour des valeurs de la colonne 'money_spent_in_ml' :
df_prep.replace(money_spent_ml_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.money_spent_in_ml.fillna(df_prep.money_spent_in_ml.median(), inplace = True)

#transformation de la variable en float64 :
df_prep.money_spent_in_ml = df_prep.money_spent_in_ml.astype(float)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.money_spent_in_ml.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.money_spent_in_ml.isna().sum(),'valeur(s) manquante(s) dans la colonne')

#Vérification du changement de type :
print('Le type de la colonne years_writting_code est :',df_prep.money_spent_in_ml.dtype)


# ## Colonne bigdata_product_used : trop de valeurs manquantes => SUPPRIMER

# In[40]:


#Suppression de la colonne : 
df_prep = df_prep.drop(['bigdata_product_used'], axis = 1)
df_prep.head()


# ## Colonne bi_tool_used : trop de valeurs manquantes => SUPPRIMER

# In[41]:


#Suppression de la colonne :
df_prep = df_prep.drop(['bi_tool_used'], axis = 1)
df_prep.head()


# ## Colonne tool_you_use_to_analyse_data : pertinent pour notre modèle de machine learning => feature (à voir)

# In[43]:


#Affichage des valeurs uniques et manquantes :
print('Les valeurs uniques sont :',df_prep.tool_you_use_to_analyse_data.unique())
print('Il y a :',df_prep.tool_you_use_to_analyse_data.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# In[44]:


#Création d'un dictionnaire pour synthétiser les valeurs de la colonne :
tool_data_analyse_dic = {'tool_you_use_to_analyse_data':{
    'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)':'bi_tool',
 'Local development environments (RStudio, JupyterLab, etc.)':'local_develp_env', 
 'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)':'cloud_software',
 'Basic statistical software (Microsoft Excel, Google Sheets, etc.)':'basic_statical_software',
 'Advanced statistical software (SPSS, SAS, etc.)':'advanced_statical_software',
    'Other':'other'}}


# Nous effectuons un retraitement des valeurs de la colonne **'tool_you_use_to_analyse_data'**, nous simplifions les intitulés des valeurs.
# 
# Cette colonne comprend **1227** valeurs manquantes (soit **21%** de données manquantes). Nous avons décidé de remplacer celle-ci par la valeur **no_info** afin de ne pas biaiser les données.

# In[46]:


#Mise à jour des valeurs de la colonne 'tool_you_use_to_analyse_data' :
df_prep.replace(tool_data_analyse_dic, inplace = True)

#Mise à jour des valeurs manquantes : 
df_prep.tool_you_use_to_analyse_data.fillna('no_info', inplace = True)

#Vérification que tout s'affiche correctement :
display(df_prep.head())
print('Liste des nouvelles valeurs uniques :',df_prep.tool_you_use_to_analyse_data.unique())

#Vérification du nombre de valeurs manquantes dans la colonne : 
print('Il y a :',df_prep.tool_you_use_to_analyse_data.isna().sum(),'valeur(s) manquante(s) dans la colonne')


# # PREPROCESSING - Etape 3 : Ensemble d'entraînement et de test : 

# Nous disposons déjà de notre variable **target** qui contient nos valeurs cibles (*Data Scientist*, *Data Analyst*, *Data Engineer*, *Machine Learning Engineer*).
# 
# Nous devons désormais **standardiser** et **centrer** les différentes valeurs de nos colonnes features.

# ## Standardisation et centralisation des valeurs numériques :

# In[47]:


#Récupération des colonnes qui contiennent des valeurs numériques uniquement : 
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df_prep_num = df_prep.select_dtypes(include=numerics)

#Vérification que tout s'affiche correctement : 
df_prep_num.head()
df_prep_num.isna().sum()


# In[48]:


#Import des librairies sklearn nécessaires à la standardisation et centralisation des données : 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[49]:


#Réinitialisation de l'index des valeurs numériques : 
df_prep_num = df_prep_num.reset_index(drop= True)


# In[50]:


#Standardisation et centralisation des données : 
df_prep_num[df_prep_num.columns] = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df_prep_num))

display(df_prep_num.head())
display(df_prep_num.shape)
df_prep_num.isna().sum()


# Les valeurs numériques sont désormais **standardisées** et **centrées**.

# ## Standardisation et centralisation des valeurs qualitatives :

# In[51]:


#Récupération des colonnes qui contiennent des valeurs numériques uniquement : 
objects = ['object']

df_prep_obj = df_prep.select_dtypes(include=objects)

#Suppression de la colonne job qui contient nos valeurs cibles : 
df_prep_obj.drop(['job'], axis = 1, inplace = True)

#Réinitialisation de l'index des valeurs qualitatives : 
df_prep_obj = df_prep_obj.reset_index(drop = True) 

#Vérification que tout s'affiche correctement : 
df_prep_obj.head()


# ### Utilisation de get_dummies : 

# In[52]:


#Transformation des données catégorielles à l'aide de get_dummies :
df_prep_dm = pd.get_dummies(df_prep_obj, dtype = np.int64)
display(df_prep_dm.head())
display(df_prep_dm.shape)


# ## Rassemblement des variables catégorielles :

# In[53]:


#Création d'un dataframe avec l'ensemble des variables catégorielles : 
features_dm = df_prep_dm.join(df_prep_num)

#Affichage de target et features_dm : 
display(target.head())
display(features_dm.head())
features_dm.isna().sum()


# Nous allons ici faire preuve d'une grande originalité et répartir nos données entre l'ensemble d'entraînement et de test sur un ratio de 80/20

# In[54]:


#Création d'un ensemble d'entraînement et de test : 
X_train, X_test,y_train, y_test = train_test_split(features_dm, target, test_size=0.2)


# ## Oversampling : 

# Oversampling aléatoire (RandomOverSampler) :  
# *Formules :* RandomOverSampler().fit_resample(X_train, y_train)
# 
# SMOTE :  
# *Formules :* X_sm, y_sm = smo.fit_resample(X_train, y_train)
# 

# In[55]:


#Importation des librairies pour l'oversampling
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import f1_score


# ### Random OverSampler :

# In[56]:


#Création d'une variable RandomOverSampler :
rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train, y_train)
print('Classes échantillon oversampled :', dict(pd.Series(y_ro).value_counts()))


# ### SMOTE :

# In[57]:


#Création d'une variable SMOTE :
smo = SMOTE()
X_sm, y_sm = smo.fit_resample(X_train, y_train)
print('Classes échantillon SMOTE :', dict(pd.Series(y_sm).value_counts()))


# ## Undersampling : 

# Undersampling aléatoire (RandomUnderSampler) :  
# *Formules :* RandomUnderSampler().fit_resample(X_train, y_train) 
# 
# Centroids :  
# *Formules :* X_cc, y_cc = cc.fit_resample(X_train, y_train)
# 

# In[58]:


#Importation de la librairies pour l'undersampling
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids


# ### Random Undersampling :

# In[59]:


#Création d'une variable Random Undersampling :
rUs = RandomUnderSampler()
X_ru, y_ru = rUs.fit_resample(X_train, y_train)
print('Classes échantillon undersampled :', dict(pd.Series(y_ru).value_counts()))


# ## Centroids :

# In[60]:


#Création d'une variable Centroids :
cc = ClusterCentroids()
X_cc, y_cc = cc.fit_resample(X_train, y_train)
print('Classes échantillon CC :', dict(pd.Series(y_cc).value_counts()))


# ## Ensemble d'entraîment pour machine learning :

# Nous disponsons désormais de deux ensembles d'entraînement pour nos tests de machine learning.
# 
# Une paire avec les méthodes d'échantillonnage en **oversampling** :  
#  - X_ro, y_ro & X_sm, y_sm
# 
# Une paire avec les méthodes d'échantillonnage en **undersampling** :
#  - X_ru, y_ru & X_cc, y_cc

# # Conclusion et suite : 

# Au travers de ce notebook, nous avons réalisé les différentes étapes de preprocessing nécessaire pour tester différents modèle de machine learning.
# 
# * Nous avons fait le choix de partir sur un jeu de donnée restreint aussi bien en terme du nombre de colonnes que de lignes. Nous nous sommes limités aux colonnes composées de questions uniques et à nos 4 métiers cibles.
# 
# * Nous avons réalisé un second tri au sein de ce jeu de données restreint en supprimant les colonnes (âge, sexe et pays) qui disposaient d'information non-pertinente pour notre modèle de machine learning.
# 
# * Nous avons également supprimé les deux colonnes : bi_tool_used and big_data_product_used, du fait qu'elles contenaient beaucoup trop de valeurs manquantes.
# 
# * Les autres colonnes qui contenaient des valeurs manquantes ont soient été remplacé par la valeur la plus fréquente dans le cas de colonnes qualitatives et par la médiane pour les colonnes quantitatives. A l'exception de la colonne tool_you_use_to_analyse_data qui avait 21% de valeurs manquantes, nous les avons ici remplacées par la valeur 'no_info'.
# 
# * Nous avons ensuite séparé les données en deux nouveaux DataFrame, l'un pour nos valeurs cibles : target et le reste dans : features_dm.
# 
# * Les données dans features_dm ont été standardisées et centrées.
# 
# * Enfin nous avons constaté que nos valeurs cibles n'étaient pas homogènes dans leurs répartitions. Nous avons donc réalisé un oversampling et undersampling de nos ensembles d'entraînement afin d'éviter tout biais du modèle de machine learning.
# 
# ### Suite :
# 
# La prochaine étape de notre projet sera de tester des modèles de machine learning de classification, en tête les modèles :
# 
# - KNN
# - Decision Tree
# - SVM
# - Random Forest

# In[61]:


#Importation des librairies pour le machine learning : 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import ensemble


# In[62]:


#Création d'une variable par modèle simple de machine learning : 
knn_m = KNeighborsClassifier()
svm_m = svm.SVC()  
dt_m = DecisionTreeClassifier()
clf = ensemble.RandomForestClassifier()


# ### Test sur X_ro, y_ro

# In[63]:


#Test avec ensemble oversamplé : 
#Score Knn : 
knn_m.fit(X_ro, y_ro)
print("score du Knn : {}".format(knn_m.score(X_test, y_test)))

#Score SVM :
svm_m.fit(X_ro, y_ro)
print("score du SVM : {}".format(svm_m.score(X_test, y_test)))

#Score Decision Tree :
dt_m.fit(X_ro, y_ro)
print("score du DT : {}".format(dt_m.score(X_test, y_test)))

#Score Random Forest :
clf.fit(X_ro, y_ro)
print("score du RF : {}".format(clf.score(X_test, y_test)))


# ### Test sur X_sm, y_sm

# In[64]:


#Test avec ensemble oversamplé : 
#Score Knn : 
knn_m.fit(X_sm, y_sm)
print("score du Knn : {}".format(knn_m.score(X_test, y_test)))

#Score SVM :
svm_m.fit(X_sm, y_sm)
print("score du SVM : {}".format(svm_m.score(X_test, y_test)))

#Score Decision Tree :
dt_m.fit(X_sm, y_sm)
print("score du DT : {}".format(dt_m.score(X_test, y_test)))

#Score Random Forest :
clf.fit(X_sm, y_sm)
print("score du RF : {}".format(clf.score(X_test, y_test)))


# ### Test sur X_ru, y_ru

# In[60]:


#Test avec ensemble undersamplé : 
#Score Knn : 
knn_m.fit(X_ru, y_ru)
print("score du Knn : {}".format(knn_m.score(X_test, y_test)))

#Score SVM :
svm_m.fit(X_ru, y_ru)
print("score du SVM : {}".format(svm_m.score(X_test, y_test)))

#Score Decision Tree :
dt_m.fit(X_ru, y_ru)
print("score du DT : {}".format(dt_m.score(X_test, y_test)))

#Score Random Forest :
clf.fit(X_ru, y_ru)
print("score du RF : {}".format(clf.score(X_test, y_test)))


# ### Test sur X_cc, y_cc

# In[66]:


#Test avec ensemble undersamplé : 
#Score Knn : 
knn_m.fit(X_cc, y_cc)
print("score du Knn : {}".format(knn_m.score(X_test, y_test)))

#Score SVM :
svm_m.fit(X_cc, y_cc)
print("score du SVM : {}".format(svm_m.score(X_test, y_test)))

#Score Decision Tree :
dt_m.fit(X_cc, y_cc)
print("score du DT : {}".format(dt_m.score(X_test, y_test)))

#Score Random Forest :
clf.fit(X_cc, y_cc)
print("score du RF : {}".format(clf.score(X_test, y_test)))


# ### Premières observations / conclusions : 
# 
# Nous venons de tester nos jeux d'entrainement oversamplé et undersamplé sur nos modèles machines learning sans hyper paramètres. Nous pouvons constater les choses suivantes : 
# 
# * Les modèles SVM & Random Forest sont ceux qui offrent les meilleurs scores peu importe l'oversampling ou undersampling choisie, à l'exception de l'undersampling Centroids où c'est le modèle KNN qui obtient le meilleur score. 
# 
# 

# ## Recherche des HyperParamètres : SVM

# Nous allons dans cette partie utiliser la fonction **GridSearchCV** afin de tester plusieurs hyper paramètres (HPs) en une fois. Cela nous permettra d'identifier les meilleurs **HPs** et ainsi d'augmenter le score du modèle **SVM**.

# La syntaxe du **GridSearchCV** se décline comme suit : 
# 
# ### Pour le KNN
# 
# #### Définition des paramètres
# knn = KNeighborsClassifier()  
# k_range = list(range(2, 20))
# metrics_list = ['cityblock','cosine','euclidean','l1','l2','manhattan','nan_euclidean']
# 
# 
# #### utilisation d'un gridsearchcv
# grid_knn = GridSearchCV(knn, param_grid_knn)
#   
# #### Fit du modèle sur X_train et y_train
# grid_knn_fit=grid_knn.fit(X_ro, y_ro)
# 
# #### Affichage du meilleur paramètre, estimator et score
# print("Le meilleure score obtenu avec les KNN est de :",grid_knn_fit.best_score_)  
# print("Le meilleure estimator obtenu avec les KNN est :",grid_knn_fit.best_estimator_)  
# print("Le meilleure paramètre obtenu avec les KNN est :",grid_knn_fit.best_params_)
# 
# #### Utilisation des meilleurs paramètre avec le modèle KNN
# knn = neighbors.KNeighborsClassifier(n_neighbors=2,metric='minkowski')  
# knn.fit(X_ro, y_ro)  
# y_pred = knn.predict(X_test)
# 
# #### Matrice de confusion
# pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
# 
# 
# ### Pour le SVM : 
# 
# #### Définition des paramètres
# param_grid_svm = [{'C': (50 , 70), 'kernel': ['rbf']},{'C' : (50 , 70), 'kernel': ['linear'] }, {'C' : (50 , 70), 'kernel': ['poly']}]
# 
#   
# #### Utilisation d'un gridsearchcv   
# grid_svm = GridSearchCV(svm.SVC(), param_grid_svm)
#   
# #### Fit du modèle sur X_train et y_train
# grid_svm_fit = grid_svm.fit(X_ro, y_ro)
# 
# #### Affichage du meilleur paramètre, estimator et score
# print(pd.DataFrame(grid_svm.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
# print("Le meilleur score obtenu avec le SVM est :",grid_svm_fit.best_score_)  
# print("Le meilleur estimator obtenu avec le SVM est :",grid_svm_fit.best_estimator_)  
# print("Le meilleur paramètre obtenu avec le SVM est :",grid_svm_fit.best_params_)
# 
# 
# ### Pour le DT : 
# 
# #### Définition des paramètres : 
# tree_param = {'criterion':['gini','entropy','log_loss'],'max_depth':np.arrange(15,26)}
# 
# #### Utilisation d'un gridsearchcv :  
# grid_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)
# 
# #### Fit du modèle sur X_train et y_train :
# grid_dt_fit = grid_dt.fit(X_ro, y_ro)
# 
# #### Affichage du meilleur paramètre, estimator et score :
# print(pd.DataFrame(grid_dt.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
# print("Le meilleur score obtenu avec le SVM est :",grid_dt_fit.best_score_)  
# print("Le meilleur estimator obtenu avec le SVM est :",grid_dt_fit.best_estimator_)  
# print("Le meilleur paramètre obtenu avec le SVM est :",grid_dt_fit.best_params_)
# 
# 
# ### Pour le RF : 
# 
# #### Définition des paramètres  
# rf = RandomForestClassifier()  
# param_grid_rf = [{'min_samples_split': [2, 31, 2],'max_features': ['sqrt', 'log2']}]
# 
# #### Utilisation d'un gridsearchcv :  
# grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, refit=True)
# 
# #### Fit le modèle sur X_train et y_train :  
# grid_rf_fit = grid_rf.fit(X_train, y_train)
# 
# #### Affichage du meilleur paramètre, estimator et score :  
# print(pd.DataFrame(grid_rf.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
# print("Le meilleur score obtenu avec la RF est :",grid_rf_fit.best_score_)  
# print("Le meilleur estimator obtenu avec la RF est :",grid_rf_fit.best_estimator_)  
# print("Le meilleur paramètre obtenu avec la RF est :",grid_rf_fit.best_params_)
# 

# In[68]:


#Import du GridSearchCV : 
from sklearn.model_selection import GridSearchCV
import chime #il s'agit d'un module qui génère un son pour prévenir de la fin d'une tâche. Pour l'installer : pip install chime


# ## SVM : 

# ### Test sur X_ro, y_ro

# In[70]:


#Définition des paramètres
param_grid_svm = [{'C': (50 , 70), 'kernel': ['rbf']},{'C' : (50 , 70), 'kernel': ['linear'] }, {'C' : (50 , 70), 'kernel': ['poly']}]
  
#Utilisation d'un gridsearchcv   
grid_svm = GridSearchCV(svm.SVC(), param_grid_svm)
  
#Fit du modèle sur X_train et y_train
grid_svm_fit = grid_svm.fit(X_ro, y_ro)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_svm.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
print("Le meilleur score obtenu avec le SVM est :",grid_svm_fit.best_score_)
print("Le meilleur estimator obtenu avec le SVM est :",grid_svm_fit.best_estimator_)
print("Le meilleur paramètre obtenu avec le SVM est :",grid_svm_fit.best_params_)

chime.error()


# ### Test sur X_sm, y_sm

# In[72]:


#Définition des paramètres
param_grid_svm = [{'C': (50 , 70), 'kernel': ['rbf']},{'C' : (50 , 70), 'kernel': ['linear'] }, {'C' : (50 , 70), 'kernel': ['poly']}]
  
#Utilisation d'un gridsearchcv   
grid_svm = GridSearchCV(svm.SVC(), param_grid_svm)
  
#Fit du modèle sur X_train et y_train
grid_svm_fit = grid_svm.fit(X_sm, y_sm)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_svm.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
print("Le meilleur score obtenu avec le SVM est :",grid_svm_fit.best_score_)
print("Le meilleur estimator obtenu avec le SVM est :",grid_svm_fit.best_estimator_)
print("Le meilleur paramètre obtenu avec le SVM est :",grid_svm_fit.best_params_)

chime.error()


# ### Test sur X_ru, y_ru

# In[74]:


#Définition des paramètres
param_grid_svm = [{'C': (50 , 70), 'kernel': ['rbf']},{'C' : (50 , 70), 'kernel': ['linear'] }, {'C' : (50 , 70), 'kernel': ['poly']}]
  
#Utilisation d'un gridsearchcv   
grid_svm = GridSearchCV(svm.SVC(), param_grid_svm)
  
#Fit du modèle sur X_train et y_train
grid_svm_fit = grid_svm.fit(X_ru, y_ru)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_svm.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
print("Le meilleur score obtenu avec le SVM est :",grid_svm_fit.best_score_)
print("Le meilleur estimator obtenu avec le SVM est :",grid_svm_fit.best_estimator_)
print("Le meilleur paramètre obtenu avec le SVM est :",grid_svm_fit.best_params_)

chime.error()


# ### Test sur X_cc, y_cc

# In[75]:


#Définition des paramètres
param_grid_svm = [{'C': (50 , 70), 'kernel': ['rbf']},{'C' : (50 , 70), 'kernel': ['linear'] }, {'C' : (50 , 70), 'kernel': ['poly'] }]
  
#Utilisation d'un gridsearchcv   
grid_svm = GridSearchCV(svm.SVC(), param_grid_svm)
  
#Fit du modèle sur X_train et y_train
grid_svm_fit = grid_svm.fit(X_cc, y_cc)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_svm.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
print("Le meilleur score obtenu avec le SVM est :",grid_svm_fit.best_score_)
print("Le meilleur estimator obtenu avec le SVM est :",grid_svm_fit.best_estimator_)
print("Le meilleur paramètre obtenu avec le SVM est :",grid_svm_fit.best_params_)

chime.error()


# In[76]:


#Importer la classe classification_report
from sklearn.metrics import classification_report


# ### SCORE SVM :
# **X_ro, y_ro : 69%**  
# X_sm, y_sm : 62%  
# X_ru, y_ru : 42%  
# X_cc, y_cc : 56%

# ### Matrice de confusion sur le meilleur résultat : 

# In[80]:


#Utilisation des meilleurs paramètres avec le modèle SVM
svm_search = svm.SVC(C= 70,kernel= 'rbf')
svm_search.fit(X_ro, y_ro)
y_pred = svm_search.predict(X_test)


#Calcul et affichage de classification_report
print( classification_report(y_test, y_pred) )


#Matrice de confusion
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# ## Arbre de Décision : 

# ### Test sur X_ro, y_ro : 

# In[81]:


#Définition des paramètres : 
tree_param = {'criterion':['gini','entropy','log_loss'],'max_depth':np.arange(15,26)}

#Utilisation d'un gridsearchcv   
grid_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)

#Fit du modèle sur X_train et y_train
grid_dt_fit = grid_dt.fit(X_ro, y_ro)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_dt.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec le DT est :",grid_dt_fit.best_score_)  
print("Le meilleur estimator obtenu avec le DT est :",grid_dt_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec le DT est :",grid_dt_fit.best_params_)

chime.error()


# ### Test sur X_sm, y_sm : 

# In[82]:


#Définition des paramètres : 
tree_param = {'criterion':['gini','entropy','log_loss'],'max_depth':np.arange(15,26)}

#Utilisation d'un gridsearchcv   
grid_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)

#Fit du modèle sur X_train et y_train
grid_dt_fit = grid_dt.fit(X_sm, y_sm)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_dt.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec le DT est :",grid_dt_fit.best_score_)  
print("Le meilleur estimator obtenu avec le DT est :",grid_dt_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec le DT est :",grid_dt_fit.best_params_)

chime.error()


# ### Test sur X_ru, y_ru : 

# In[83]:


#Définition des paramètres : 
tree_param = {'criterion':['gini','entropy','log_loss'],'max_depth':np.arange(15,26)}

#Utilisation d'un gridsearchcv   
grid_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)

#Fit du modèle sur X_train et y_train
grid_dt_fit = grid_dt.fit(X_ru, y_ru)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_dt.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec le DT est :",grid_dt_fit.best_score_)  
print("Le meilleur estimator obtenu avec le DT est :",grid_dt_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec le DT est :",grid_dt_fit.best_params_)

chime.error()


# ### Test sur X_cc, y_cc : 

# In[84]:


#Définition des paramètres : 
tree_param = {'criterion':['gini','entropy','log_loss'],'max_depth':np.arange(15,26)}

#Utilisation d'un gridsearchcv   
grid_dt = GridSearchCV(DecisionTreeClassifier(), tree_param)

#Fit du modèle sur X_train et y_train
grid_dt_fit = grid_dt.fit(X_cc, y_cc)

#Affichage du meilleur paramètre, estimator et score
print(pd.DataFrame(grid_dt.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec le DT est :",grid_dt_fit.best_score_)  
print("Le meilleur estimator obtenu avec le DT est :",grid_dt_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec le DT est :",grid_dt_fit.best_params_)

chime.error()


# ### SCORE Arbre de décision :
# **X_ro, y_ro : 78%**  
# X_sm, y_sm : 56%  
# X_ru, y_ru : 35%  
# X_cc, y_cc : 49%

# ### Matrice de confusion sur le meilleur résultat : 

# In[85]:


#Utilisation des meilleurs paramètres avec le modèle Decision Tree :
dt_search = DecisionTreeClassifier(criterion = 'entropy', max_depth = 25)
dt_search.fit(X_ro, y_ro)
y_pred = dt_search.predict(X_test)

#Calcul et affichage de classification_report
print( classification_report(y_test, y_pred) )

#Matrice de confusion
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# ## Random Forest :

# ### Test sur X_ro, y_ro :

# In[86]:


#Définition des paramètres  
rf = ensemble.RandomForestClassifier()  
param_grid_rf = [{'min_samples_split': [2, 31, 2],'max_features': ['sqrt', 'log2']}]

#Utilisation d'un gridsearchcv :  
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, refit=True)

#Fit le modèle sur X_train et y_train :  
grid_rf_fit = grid_rf.fit(X_ro, y_ro)

#Affichage du meilleur paramètre, estimator et score :  
print(pd.DataFrame(grid_rf.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec la RF est :",grid_rf_fit.best_score_)  
print("Le meilleur estimator obtenu avec la RF est :",grid_rf_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec la RF est :",grid_rf_fit.best_params_)

chime.error()


# ### Test sur X_sm, y_sm :

# In[87]:


#Définition des paramètres  
rf = ensemble.RandomForestClassifier()  
param_grid_rf = [{'min_samples_split': [2, 31, 2],'max_features': ['sqrt', 'log2']}]

#Utilisation d'un gridsearchcv :  
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, refit=True)

#Fit le modèle sur X_train et y_train :  
grid_rf_fit = grid_rf.fit(X_sm, y_sm)

#Affichage du meilleur paramètre, estimator et score :  
print(pd.DataFrame(grid_rf.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec la RF est :",grid_rf_fit.best_score_)  
print("Le meilleur estimator obtenu avec la RF est :",grid_rf_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec la RF est :",grid_rf_fit.best_params_)

chime.error()


# ### Test sur X_ru, y_ru :

# In[88]:


#Définition des paramètres  
rf = ensemble.RandomForestClassifier()  
param_grid_rf = [{'min_samples_split': [2, 31, 2],'max_features': ['sqrt', 'log2']}]

#Utilisation d'un gridsearchcv :  
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, refit=True)

#Fit le modèle sur X_train et y_train :  
grid_rf_fit = grid_rf.fit(X_ru, y_ru)

#Affichage du meilleur paramètre, estimator et score :  
print(pd.DataFrame(grid_rf.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec la RF est :",grid_rf_fit.best_score_)  
print("Le meilleur estimator obtenu avec la RF est :",grid_rf_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec la RF est :",grid_rf_fit.best_params_)

chime.error()


# ### Test sur X_cc, y_cc :

# In[90]:


#Définition des paramètres  
rf = ensemble.RandomForestClassifier()  
param_grid_rf = [{'min_samples_split': [2, 31, 2],'max_features': ['sqrt', 'log2']}]

#Utilisation d'un gridsearchcv :  
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, refit=True)

#Fit le modèle sur X_train et y_train :  
grid_rf_fit = grid_rf.fit(X_cc, y_cc)

#Affichage du meilleur paramètre, estimator et score :  
print(pd.DataFrame(grid_rf.cv_results_)[['params', 'mean_test_score', 'std_test_score']])  
print("Le meilleur score obtenu avec la RF est :",grid_rf_fit.best_score_)  
print("Le meilleur estimator obtenu avec la RF est :",grid_rf_fit.best_estimator_)  
print("Le meilleur paramètre obtenu avec la RF est :",grid_rf_fit.best_params_)

chime.error()


# ### SCORE Random Forest :
# **X_ro, y_ro : 80%**  
# X_sm, y_sm : 68%  
# X_ru, y_ru : 43%  
# X_cc, y_cc : 56%

# ### Matrice de confusion sur le meilleur résultat : 

# In[91]:


#Utilisation des meilleurs paramètres avec le modèle RF
rf = ensemble.RandomForestClassifier(max_features = 'log2', min_samples_split = 2)
rf.fit(X_ro, y_ro)
y_predict = rf.predict(X_test)

#Calcul et affichage de classification_report
print( classification_report(y_test, y_predict) )

#Matrice de confusion
pd.crosstab(y_test, y_predict, rownames=['Classe réelle'], colnames=['Classe prédite'])


# ## KNN :

# ### Test sur X_ro, y_ro :

# In[97]:


knn = KNeighborsClassifier()
k_range = list(range(2, 20))
metrics_list = ['cityblock','cosine','euclidean','l1','l2','manhattan','nan_euclidean']
param_grid_knn = dict(n_neighbors=k_range, metric = metrics_list)

#Utilisation d'un gridsearchcv
grid_knn = GridSearchCV(knn, param_grid_knn)

#Fit du modèle sur X_train et y_train
grid_knn_fit=grid_knn.fit(X_ro, y_ro)

#Affichage du meilleur paramètre, estimator et score
print("Le meilleure score obtenu avec les KNN est de :",grid_knn_fit.best_score_)
print("Le meilleure estimator obtenu avec les KNN est :",grid_knn_fit.best_estimator_)
print("Le meilleure paramètre obtenu avec les KNN est :",grid_knn_fit.best_params_)

chime.error()


# ### Test sur X_sm, y_sm :

# In[98]:


knn = KNeighborsClassifier()
k_range = list(range(2, 20))
metrics_list = ['cityblock','cosine','euclidean','l1','l2','manhattan','nan_euclidean']
param_grid_knn = dict(n_neighbors=k_range, metric = metrics_list)

#Utilisation d'un gridsearchcv
grid_knn = GridSearchCV(knn, param_grid_knn)

#Fit du modèle sur X_train et y_train
grid_knn_fit=grid_knn.fit(X_sm, y_sm)

#Affichage du meilleur paramètre, estimator et score
print("Le meilleure score obtenu avec les KNN est de :",grid_knn_fit.best_score_)
print("Le meilleure estimator obtenu avec les KNN est :",grid_knn_fit.best_estimator_)
print("Le meilleure paramètre obtenu avec les KNN est :",grid_knn_fit.best_params_)

chime.error()


# ### Test sur X_ru, y_ru :

# In[99]:


knn = KNeighborsClassifier()
k_range = list(range(2, 20))
metrics_list = ['cityblock','cosine','euclidean','l1','l2','manhattan','nan_euclidean']
param_grid_knn = dict(n_neighbors=k_range, metric = metrics_list)

#Utilisation d'un gridsearchcv
grid_knn = GridSearchCV(knn, param_grid_knn)

#Fit du modèle sur X_train et y_train
grid_knn_fit=grid_knn.fit(X_ru, y_ru)

#Affichage du meilleur paramètre, estimator et score
print("Le meilleure score obtenu avec les KNN est de :",grid_knn_fit.best_score_)
print("Le meilleure estimator obtenu avec les KNN est :",grid_knn_fit.best_estimator_)
print("Le meilleure paramètre obtenu avec les KNN est :",grid_knn_fit.best_params_)

chime.error()


# ### Test sur X_cc, y_cc :

# In[100]:


knn = KNeighborsClassifier()
k_range = list(range(2, 20))
metrics_list = ['cityblock','cosine','euclidean','l1','l2','manhattan','nan_euclidean']
param_grid_knn = dict(n_neighbors=k_range, metric = metrics_list)

#Utilisation d'un gridsearchcv
grid_knn = GridSearchCV(knn, param_grid_knn)

#Fit du modèle sur X_train et y_train
grid_knn_fit=grid_knn.fit(X_cc, y_cc)

#Affichage du meilleur paramètre, estimator et score
print("Le meilleure score obtenu avec les KNN est de :",grid_knn_fit.best_score_)
print("Le meilleure estimator obtenu avec les KNN est :",grid_knn_fit.best_estimator_)
print("Le meilleure paramètre obtenu avec les KNN est :",grid_knn_fit.best_params_)

chime.error()


# ### SCORE KNN :
# **X_ro, y_ro : 69%**  
# X_sm, y_sm : 66%  
# X_ru, y_ru : 39%  
# X_cc, y_cc : 49%

# ### Matrice de confusion sur le meilleur résultat : 

# In[102]:


#Utilisation des meilleurs paramètre avec le modèle KNN
knn = KNeighborsClassifier(n_neighbors=2,metric='cosine')
knn.fit(X_ro, y_ro)
y_pred = knn.predict(X_test)


#Calcul et affichage de classification_report
print( classification_report(y_test, y_pred))

#Matrice de confusion
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# ## Observations / Conclusions :

# * Il semble que nous obtenions de meilleurs résultats avec la méthode d'oversampling Random OverSampler (RO).
# 
# * Les deux meilleurs modèles de machines learning pour la classification de nos 4 classes sont :
#  * Le Decision Tree avec un score de : 78%
#  * La Random Forest avec un score de : 79%
#  
# Néanmoins ces bons résultats sont à pondérer, en effet, lorsque nous réalisons une matrice de confusion, nous observons les comportements suivants : 
# 
# * Les modèles n'arrivent pas à classer correctement les différentes classes. 
# 
# * Seuls les classes Data Analyst et Data Scientist arrivent à être identifié de façon significative.
# 
# * Les classes Data Engineer et Machine Learning Engineer sont très mal identifiées par les modèles de machine learning.
# 
# Il sera nécessaire dans de prochaine itération d'améliorer encore la sélection des critères (features) et se concentrer sur les modèles Decision Tree et Random Forest sur la méthode d'oversampling Random OverSampler (RO).
# 
# Remarques : 
# * Le modèle de machine learning **Decision Tree** est sensible à **l'overfitting**, ce qui peut venir fausser les résultats obtenus.
# 
# * Le modèle de machine learning **Random Forest** est considéré comme une **'black box'** (boîte noire) ce qui rend difficile l'interprétation/compréhension de son fonctionnement.
