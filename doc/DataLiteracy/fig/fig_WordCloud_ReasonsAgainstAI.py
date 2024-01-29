#!/usr/bin/env python
# coding: utf-8

# # Data Literacy
# #### University of Tübingen, Winter 2023/24
# ## Generative AI in student learning (analysis)
# 
# &copy; Jan Goebel, 2024. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

# In[1]:


# imports for data anaylsis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pingouin as pg
import matplotlib.cm as cm
from scipy.stats import pearsonr
from tueplots import bundles
from tueplots.constants.color import rgb


# In[2]:


# load survey data from 23rd of january, 2024
df_full = pd.read_csv('SurveyResults_23_01_2024.csv') # full CSV data with question codes
df_excel = pd.read_excel('Processed_Recoded_dat.xlsx') # modified Excel data with summarised universities and subjects

df_full.head(10)


# ### In case of font error use the following code
# 
# import matplotlib as mpl
# import shutil
# 
# ##### Get the cache directory
# cache_dir = mpl.get_cachedir()
# 
# ##### Remove the entire directory
# shutil.rmtree(cache_dir)
# 

# In[3]:


df_full.shape


# In[4]:


# sort dfs by column id to replace the in excel modified columns in the csv df
df_full = df_full.sort_values(by='id')
df_excel = df_excel.sort_values(by='id')

# Rename column in df_full
df_full.rename(columns={'ReasonsAgainstAI': 'ReasonsAgainstAI_translated'}, inplace=True)

# Replace modified columns in df_full
df_full['Subject'] = df_excel['Subject']
df_full['University'] = df_excel['University']
df_full['AmoutUniAI'] = df_excel['AmoutUniAI']
df_full['ReasonsAgainstAI_translated'] = df_excel['ReaonsAgainstAI_translated']

# Show modified columns
df_full[['Subject', 'University', 'AmoutUniAI', 'ReasonsAgainstAI_translated']].head(5)


# In[5]:


# Exclude people that stopped the questionnaire immideatly after the 
# the personal information or before or are no students

df_relevant = df_full.drop(df_full[df_full['CutOff'] == 'Nein (No)'].index)
df_relevant = df_relevant.drop(df_relevant[df_relevant['lastpage'] == 1].index)
df_relevant = df_relevant.dropna(subset=['lastpage'])
df_relevant.shape


# In[6]:


# Total of N=46 (57%) responses

# Set plotting style for ICML 2022
plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=False))

# Feedback translated into english using the deepl translator
# Selecting the relevant column
text = df_relevant['ReasonsAgainstAI_translated'].dropna()

# Convert column values to a single string
text = " ".join(map(str, text))

# Preprocess text using nltk
# Tokenize text into words using the default word tokenizer
words = nltk.word_tokenize(text)
# Remove stopwords
stopwords = nltk.corpus.stopwords.words("english")
# Add the word "n't" to the list
stopwords.append("n't")
stopwords.append("e")
words = [word for word in words if word not in stopwords]
# Join words back into a single string
text = " ".join(words)

# Create word cloud object and generate word cloud
wc = WordCloud(max_words=50, background_color="white", colormap="viridis")
wc.generate(text)

# Plot word cloud using matplotlib
plt.figure(figsize=(3, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
# Add a title to the plot
#plt.title("Your Title")

# Save the figure
plt.savefig('fig_WordCloud_ReasonsAgainstAI.pdf')

plt.show()

