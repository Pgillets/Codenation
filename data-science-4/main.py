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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv", decimal = ',')


# In[4]:


countries.head(5)


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.info()
countries.describe()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[22]:


countries_ajusted = countries.copy()

string_columns = countries_ajusted.columns[countries_ajusted.dtypes == object]

countries_ajusted[string_columns] = countries_ajusted[string_columns].apply(lambda x: x.str.strip())


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[7]:


unique_regions = list(countries_ajusted['Region'].unique())
unique_regions.sort()
unique_regions


# In[8]:


def q1():
    return unique_regions


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[18]:


KBinsDiscretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
KBinsDiscretizer.fit(countries_ajusted[['Pop_density']])

pop_discret = KBinsDiscretizer.transform(countries_ajusted[['Pop_density']])

quantile_90 = np.quantile(pop_discret,0.9)


# In[19]:


int(sum(pop_discret > quantile_90))


# In[20]:


def q2():
    return int(sum(pop_discret > quantile_90))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[23]:


countries_ajusted['Climate'] = countries_ajusted['Climate'].fillna(countries_ajusted['Climate'].mean())
countries_ajusted['Climate'].isna().sum()


# In[29]:


one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
one_hot_encoder.fit(countries_ajusted[['Region', 'Climate']])
onde_hot_encoded_shape = one_hot_encoder.transform(countries_ajusted[['Region', 'Climate']]).shape
onde_hot_encoded_shape


# In[30]:


def q3():
    return onde_hot_encoded_shape[1]


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[12]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[34]:


num_pipeline = Pipeline(steps=[('Imputer', SimpleImputer(strategy='median')),('StandardScaler', StandardScaler())])

int64_float64_columns = countries_ajusted.columns[(countries_ajusted.dtypes == 'int64') | (countries_ajusted.dtypes == float)]

pipeline_transformation = num_pipeline.fit_transform(countries_ajusted[int64_float64_columns])

test_country_pipeline = num_pipeline.transform([test_country[2:]])

new_arable = float(np.round((test_country_pipeline[0,9]),3))
new_arable


# In[35]:


def q4():
    return new_arable


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[42]:


quantil1 = countries_ajusted['Net_migration'].quantile(0.25)
quantil3 = countries_ajusted['Net_migration'].quantile(0.75)
iqr = quantil3 - quantil1

iqr_limits = [quantil1-(1.5*iqr), quantil3+(1.5*iqr)]

outliers_high = sum(countries_ajusted['Net_migration'] > iqr_limits[1])
outliers_low = sum(countries_ajusted['Net_migration'] < iqr_limits[0])

resposta5 = (outliers_low, outliers_high, False)
resposta5


# In[43]:


def q5():
    return resposta5


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

# In[46]:


categorias = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categorias, shuffle=True, random_state=42)


# In[48]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)

phone_word_count = int(newsgroups_counts[:,count_vectorizer.vocabulary_['phone']].sum())
phone_word_count


# In[49]:


def q6():
    return phone_word_count


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[51]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(newsgroups.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups.data)

phone_tf_idf = float(np.round(newsgroups_tfidf_vectorized[:, tfidf_vectorizer.vocabulary_['phone']].sum(),3))
phone_tf_idf


# In[52]:


def q7():
    return phone_tf_idf

