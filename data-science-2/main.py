#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[287]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import pandas.util.testing as tm
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# In[288]:


# %matplotlib inline

from IPython.core.pylabtools import figsize

figsize(20, 6)

sns.set()


# In[289]:


athletes = pd.read_csv(r'data/athletes.csv')


# In[290]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[291]:


athletes.sample(5)


# In[292]:


athletes.info()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[293]:


height = get_sample(athletes, 'height', n=3000)
weight = get_sample(athletes, 'weight', n=3000)
ALPHA = 0.05


# In[294]:


def q1():
    stats, p_value = sct.shapiro(height)

    return  bool(p_value > ALPHA)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[295]:


fig, ax =plt.subplots(1,2)
sns.distplot(height, bins= 25, hist_kws={'density':True}, ax=ax[0]);
sm.qqplot(height, fit=True, line='45', alpha=.5, ax=ax[1]);
fig.show()


# Em uma primeira análise é possível observar que forma do gráfico apresenta uma simetria (formato que se assemelha a um sino), além disso, é possível observar no diagrama de probabilidade normal, que apesar de alguns dados apresentarem uma pequena variação nas extremidades em sua maioria eles econtram-se alinhados, o que justificaria ratificar a normalidade da distribuição. 
# 
# No entanto no teste de Shapiro-Wilk constatamos que o valor-p é menor que $\alpha$ e logo rejeitamos $H_0$, classificando a distribuição desses dados como sendo uma distribuição não normal. 
# 
# Uma provável forma de conseguir outro resultado é limitar o tamanho da amostra.

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[296]:


def q2():
    stats, p_value = sct.jarque_bera(height)

    return  bool(p_value > ALPHA)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# A idéia de utilizar o teste Jarque-Bera é validar a normalidade da distribuição considerando outros elementos. O teste Jarque-Bera utiliza como parâmetros os coeficientes de `curtose` e `assimetria` (que para a distribuição normal são de 3 e 0, respectivamente).  Queremos saber se nossa distribuição é aproximadamente normal porque, desvios muitos grandes, como, por exemplo, uma curtose acima de 4 e assimetria acima de 1 invalidaria nossos erros-padrão e intervalos de confiança.
# 
# No entando ao utilizar o teste Jarque-Bera, ainda não podemos afimar que as alturas são normalmente distribuidas e continuamos rejeitando $H_0$.

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[297]:


def q3():
    stats, p_value = sct.normaltest(weight)

    return bool(p_value > ALPHA)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[298]:


fig, ax =plt.subplots(1,2)
sns.distplot(weight, bins= 25, hist_kws={'density':True}, ax=ax[0]);
weight.plot(kind='box', sym='r.', ax=ax[1]);
fig.show()


# Através do gráfico da distribuição dos pesos é possível observar a distribuição possui uma cauda mais longa à direita, indicando a ocorrência de valores altos com baixa frequência. Esse tipo de distribuição é denominada assimétrica positiva ou à direita.
# 
# O boxplot a direita ratifica que a distribuição é assimétrica. Há uma alta concentração de dados nos valores mais baixos. A cauda mais longa da distribuição fica à direita, indicando a ocorrência de valores altos com baixa frequência destacados em vermleho. O teste de normalidade de D'Agostino-Pearson ratifica que o valor-p é menor que $\alpha$ e logo rejeitamos $H_0$, classificando a distribuição desses dados como sendo uma distribuição assimétrica ou não normal.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[299]:


def q4():
    log = np.log(weight)
    stats, p_value = sct.normaltest(log)

    return bool(p_value > ALPHA)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[300]:


fig, ax =plt.subplots(1,2)
sns.distplot(weight, bins= 25, hist_kws={'density':True}, ax=ax[0]);
sm.qqplot(weight, fit=True, line='45', alpha=.5, ax=ax[1]);
fig.show()


# A forma do gráfico e o resultado demonstram visivelmente que a forma da distribuição é assimétrica. Além disso, é possível observar no diagrama de probabilidade normal, que alguns dados apresentarem uma variação considerável nas extremidades atestando a ausênia de normalidade da distribuição.

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[301]:


bra = athletes[athletes['nationality'] == 'BRA']
usa = athletes[athletes['nationality'] == 'USA']
can = athletes[athletes['nationality'] == 'CAN']


# In[302]:


import statistics 

print(statistics.variance(bra['height'].dropna()))
print(statistics.variance(usa['height'].dropna()))
print(statistics.variance(can['height'].dropna()))


# In[303]:


def q5():
    stats, p_value = sct.ttest_ind(bra['height'], usa['height'], equal_var=False, nan_policy = 'omit')

    return bool(p_value > 0.025)


# Não podemos afirmar que as médias são estatísticamente iguais.
# 
# Um ponto que merece destaque é o parâmetro `equal_var` da função `scipy.stats.ttest_ind()`. Esse parâmetro aplica o t-test de Welch ao invés do test T-student considerando a variância das amostras. O teste t de Welch, diferentemente do teste t de Student , não tem a suposição de variância igual (no entanto, ambos os testes têm a suposição de normalidade). No entanto, quando os tamanhos e as variações das amostras são desiguais, o teste t de Student não é confiável; As tendências de Welch acabam por oferecer melhor desempenho.

# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[304]:


def q6():
    stats, p_value = sct.ttest_ind(bra['height'], can['height'], equal_var=False, nan_policy = 'omit')

    return bool(p_value >= 0.025)


# Não podemos afirmar que as médias são estatísticamente iguais.
# 

# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[305]:


def q7():
    stats, p_value = sct.ttest_ind(usa['height'], can['height'], equal_var=False, nan_policy = 'omit')

    return float(round(p_value, 8))


# In[306]:


plt.hist(usa['height'], bins= 50, alpha=0.5, label='USA Heights')
plt.axvline(usa['height'].mean(), color='r', linestyle='dashed', linewidth=1)
plt.hist(can['height'], bins= 50, alpha=0.5, label='CAN Heights')
plt.axvline(can['height'].mean(), color='g', linestyle='dashed', linewidth=1)
plt.legend(loc='upper right')
plt.show()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# Aparentemente não. Apesar do teste de hipótese apresentar um valor-p menor que o alpha definido, as médias são muito próximas quando comparadas as distribuições dos dois países. Segundo o _valor-p_ (0.00046601) o ideal seria rejeitar $H_0$.

# In[ ]:




