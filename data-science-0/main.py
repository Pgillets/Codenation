#!/usr/bin/env python
# coding: utf-8

# ## Desafio 1 -  Manipulação de Dados
# 
# O objetivo dessa semana de treinamento foi aprimorar os conhecimento em manipulação de dados utilizando a biblioteca [Pandas](https://pandas.pydata.org/docs/). Para isso, foi disponibilizado um dataset referente às transações de compras realizadas em uma loja de varejo durante a [Black Friday](https://www.kaggle.com/mehdidag/black-friday).
# 
# <img src="data.svg" width="300" height="300"/>
# 
# Neste material foram realizadas as seguintes tarefas:  
# * **Análise exploratória com o objetivo de compreender o perfil dos consumidores e o conjunto de dados;**
# * **Desenvolvimento das questão requisitadas, obtendo a pontuação máxima e demonstração da estratégia de resolução adotada;**
# * **Aprofundamento teórico na tarefa de escalonamento de características, tratando técnicas de Normalização e Padronização dos dados.**
# 

# ### Importações

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ### Informações básicas

# In[3]:


black_friday.head()


# In[4]:


black_friday.shape


# In[5]:


black_friday.isna().sum()


# ### Questão 1
# 
# * **Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.**

# In[6]:


def q1():
    return black_friday.shape


# In[7]:


q1()


# ### Questão 2
# 
# * **Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.**

# O método ***query*** nos permite realizar consultas em nosso dataframe utilizando expressões booleanas. Neste caso, irei realizar uma consulta por mulheres *(Gender == F)* com idade entre 26 e 35 anos *(Age == '26-35')*. Como retorno, teremos um dataframe que satisfaz a condição anterior, porém precisamos obter o nº de observações deste dataframe. Para isso basta utilizarmos o atributo ***shape***, acessando sua primeira posição.

# In[8]:


def q2():
    return black_friday.query("Gender == 'F' & Age == '26-35'").shape[0]


# In[9]:


q2()


# ### Questão 3
# 
# * **Quantos usuários únicos há no dataset? Responda como um único escalar.**

# In[10]:


def q3():
    return len(black_friday['User_ID'].unique())


# In[11]:


q3()


# ### Questão 4
# 
# * **Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.**

# In[12]:


def q4():
    return len(black_friday.dtypes.unique())


# In[13]:


q4() 


# ### Questão 5
# 
# * **Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.**

# In[14]:


percent_of_null = (black_friday.isnull().sum().max() / black_friday.shape[0])
percent_of_null


# In[15]:


def q5():
    return 0.6944102891306734


# In[16]:


q5()


# ### Questão 6
# 
# * **Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.**

# In[37]:


most_na = black_friday.isna().sum().max()
most_na


# In[38]:


def q6():
    return 373299


# In[39]:


q6()


# ### Questão 7
# 
# * **Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.**

# O valor mais frequente em um conjunto de dados é a moda (.mode())

# In[20]:


def q7():
    return black_friday['Product_Category_3'].mode().values[0]


# In[21]:


q7()


# ### Questão 8
# 
# * **Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.**

# Usaremos o escalonamento ***min_max*** (**normalização**). Sua função é redimensionar os valores para um intervalo específico, como 0 e 1, dessa forma todos os atributos irão possuír a mesma escala. Isso é feito subtruindo o valor pelo valor mínimo e dividindo pelo máximo menos o mínimo. 
# 
# $$
# X_{changed} = \frac{X - X_{min}}{X_{max} - X_{min}}
# $$
# 
# 

# In[22]:


min_max = MinMaxScaler()
purchase_normalized = min_max.fit_transform(black_friday['Purchase'].values.reshape(-1, 1))
purchase_normalized


# In[40]:


norm_mean = purchase_normalized.mean()
norm_mean


# In[43]:


def q8():
    return 0.3847939036269795


# In[44]:


q8()


# ### Questão 9
# 
# * **Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.**

# A padronização não vincula os dados a um intervalo específico, como no escalonamento ***min-max***. Ela subtrai o valor médio e, em seguida, divide pela variância. A padronização é feita através da fórmula *z-score*:
# 
# $$
# z = \frac{x - \mu }{\sigma}
# $$
# 
# A padronização é muito menos afetada por ***outliers*** em comparação com a normalização.  

# In[25]:


standard = StandardScaler()
purchase_standardized = standard.fit_transform(black_friday['Purchase'].values.reshape(-1, 1))
purchase_standardized


# In[26]:


len(purchase_standardized)


# In[27]:


purchase_standardized[(purchase_standardized >= -1) & (purchase_standardized <= 1)]


# In[28]:


def q9():
    return len(purchase_standardized[(purchase_standardized >= -1) & (purchase_standardized <= 1)])


# In[29]:


q9()


# ### Questão 10
# 
# * **Podemos afirmar que se uma observação é null em `Product_Category_2` ela também é em `Product_Category_3`? Responda com um bool (`True`, `False`).**

# Primeiramente, obtive um um dataframe auxiliar contendo todas as observações em que o atributo *Product_Category_2* é nulo. O método *sum()* não contabiliza valores nulos, portanto a soma de uma coluna nula será 0. Dessa forma, apenas verifiquei se a soma de ambas colunas (*Product_Category_2 e Product_Category_3*) são iguais, ou seja, 0. Se a condição for satisfeita, toda observação nula em *Product_Category_2* será nula em *Product_Category_3*.

# In[30]:


def q10():
    aux = black_friday[black_friday['Product_Category_2'].isnull()]
    return bool(aux['Product_Category_2'].sum() == aux['Product_Category_3'].sum())


# In[31]:


q10()

