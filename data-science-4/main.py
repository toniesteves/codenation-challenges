#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

import pandas.util.testing as tm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv('countries.csv', decimal=',')


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[5]:


countries['Region'] = countries['Region'].apply(lambda x: x.strip())
countries.shape


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
countries.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[7]:


def q1():
    return sorted(countries['Region'].unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[8]:


k_bins_discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

discrete_pop_density = k_bins_discretizer.fit_transform(countries['Pop_density'].values.reshape(-1, 1))


# In[9]:


def q2():
    return int((discrete_pop_density==9).sum())


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


countries[['Region','Climate']].info()


# In[11]:


one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)

data_encoded = one_hot_encoder.fit_transform(countries[['Region','Climate']].fillna({'Climate': 0}))


# In[12]:


def q3():
    return data_encoded.shape[1]


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[13]:


test_country = [
    'Test Country', 'NEAR EAST', 
    -0.19032480757326514, -0.3232636124824411, 
    -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 
    1.0119784924248225, 0.6189182532646624, 
    1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 
    1.3163604645710438, -0.3699637766938669, 
    -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[14]:


test_data = pd.DataFrame([test_country], columns=countries.columns)


# In[15]:


data_features = countries.select_dtypes('number').columns

data_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[('num', data_pipeline, data_features)], 
    remainder='drop'
)
    
preprocessor.fit(countries)


# In[16]:


def q4():
    arable = preprocessor.transform(test_data)[0][data_features.get_loc('Arable')]
    return float(round(arable, 3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[17]:


net_migration = countries['Net_migration'].dropna()


# In[18]:


# figsize(12, 6)
net_migration.plot(kind='box', sym='r.', vert=False);


# In[19]:


def q5():
    
    q1, q3 = net_migration.quantile([.25, .75])
    iqr = q3 - q1

    outliers_down = int(net_migration[net_migration < q1 - 1.5*iqr].shape[0])
    outliers_up = int(net_migration[net_migration > q3 + 1.5*iqr].shape[0])
    threshold = bool((outliers_down + outliers_up)/net_migration.shape[0] < 0.1)
    
    return (outliers_down, outliers_up, threshold)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[37]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


# In[39]:


def q6():
    vectorizer = CountVectorizer().fit(newsgroup.data)
    data = vectorizer.transform(newsgroup.data)
    
    return int(data[:, vectorizer.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[23]:


vectorizer = TfidfVectorizer().fit(newsgroup.data)
data = vectorizer.transform(newsgroup.data)


# In[24]:


def q7():
    return float(data[:, vectorizer.vocabulary_['phone']].sum().round(3))

