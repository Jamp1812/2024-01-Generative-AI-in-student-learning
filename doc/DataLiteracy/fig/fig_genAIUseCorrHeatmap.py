#!/usr/bin/env python
# coding: utf-8

# # Data Literacy
# #### University of Tübingen, Winter 2023/24
# ## Generative AI in student learning (analysis)
# 
# &copy; Jan Goebel, 2024. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

# In[17]:


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


# In[18]:


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

# In[19]:


df_full.shape


# In[20]:


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


# In[21]:


# Exclude people that stopped the questionnaire immideatly after the 
# the personal information or before or are no students

df_relevant = df_full.drop(df_full[df_full['CutOff'] == 'Nein (No)'].index)
df_relevant = df_relevant.drop(df_relevant[df_relevant['lastpage'] == 1].index)
df_relevant = df_relevant.dropna(subset=['lastpage'])
df_relevant.shape


# In[22]:


# Create mappings for further likert-scale questions

# List of fear of AI items
ai_fear = ['FearOfAI[SQ001]', 'FearOfAI[SQ002]', 'FearOfAI[SQ003]']

# List of futher AI items
ai_further = ['FurtherAI[SQ001]', 'FurtherAI[SQ002]', 'FurtherAI[SQ003]',
             'FurtherAI[SQ004]']

# Define the mapping of responses to codes
response_mapping2 = {
    "trifft voll zu": 2,
    "trifft eher zu": 1,
    "teils teils": 0,
    "trifft eher nicht zu": -1,
    "trifft gar nicht zu.": -2
}

# Replace the responses with codes in the fear of AI columns
df_relevant[ai_fear] = df_relevant[ai_fear].replace(response_mapping2)

# Replace the responses with codes in the further AI item columns
df_relevant[ai_further] = df_relevant[ai_further].replace(response_mapping2)


# In[26]:


df_relevant[ai_further].head(10)


# In[23]:


# Create "Fear of AI scale value" for each person
df_relevant['Fear_of_AI_Scale'] = df_relevant[ai_fear].sum(axis=1)


# In[24]:


# Recode gender, focusing on females and males

# Define the mapping of responses to codes
gender_mapping = {
    "Männlich (Male)": 0,
    "Weiblich (Female)": 1
}

# Replace the responses with codes in the gender column
df_relevant['Gender'] = df_relevant['Gender'].replace(gender_mapping)

# Replace all other values with NaN
df_relevant['Gender'] = df_relevant['Gender'].apply(lambda x: x if x in gender_mapping.values() else np.nan)
# Recode GenAIuse

# Define the mapping of responses to codes
genAIuse_mapping = {
    "Ja (Yes)": 1,
    "Nein (No)": 0
}

# Replace the responses with codes in the genAIuse column
df_relevant['GenAIUse'] = df_relevant['GenAIUse'].replace(genAIuse_mapping)

# Create dataframe with relevant items for correlation
df_corrDat = df_relevant[['Gender', 'Age', 'Fear_of_AI_Scale', 'GenAIUse', 'FearOfAI[SQ001]', 
                         'FearOfAI[SQ002]', 'FearOfAI[SQ003]', 'FurtherAI[SQ001]', 'FurtherAI[SQ002]',
                         'FurtherAI[SQ003]', 'FurtherAI[SQ004]']].dropna()

df_corrDat.shape


# In[25]:


# Create correlation heatmap

# Create correlation df
corrDat = df_corrDat[['Gender', 'Age', 'Fear_of_AI_Scale', 'GenAIUse', 'FurtherAI[SQ001]', 'FurtherAI[SQ002]',
                         'FurtherAI[SQ003]']]

# Calculate correlation matrix
corr = corrDat.corr()

# Define custom labels
x_labels = ["Gender", "Age", "Fear of AI", "Gen AI use", "AI knowledge", "Pos impact", 
            "Neg impact"]
y_labels = ["Gender", "Age", "Fear of AI", "Gen AI use", "AI knowledge", "Pos impact", 
            "Neg impact"]

# Plot the heatmap with custom labels
ax = sns.heatmap(corr, annot=True, fmt=".1f", cmap="viridis", xticklabels=x_labels, yticklabels=y_labels)
#plt.title("Correlation Matrix")

# Get the colorbar
cbar = ax.collections[0].colorbar

# Define the tick locations and labels
tick_locs = np.linspace(start=-0.5, stop=1.0, num=7)  # Adjust these values according to your needs
tick_labels = ['-0.50', '-0.25', '0.0', '0.25', '0.50', '0.75', '1.0']

# Set the ticks and labels on the colorbar
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(tick_labels)

# Save the figure
#plt.savefig('Heatmap_genAI_corr.pdf')

plt.show()

