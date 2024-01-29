#!/usr/bin/env python
# coding: utf-8

# # Data Literacy
# #### University of Tübingen, Winter 2023/24
# ## Generative AI in student learning (analysis)
# 
# &copy; Jan Goebel, 2024. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

# In[2]:


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


# In[3]:


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

# In[4]:


df_full.shape


# In[5]:


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


# In[6]:


# Exclude people that stopped the questionnaire immideatly after the 
# the personal information or before or are no students

df_relevant = df_full.drop(df_full[df_full['CutOff'] == 'Nein (No)'].index)
df_relevant = df_relevant.drop(df_relevant[df_relevant['lastpage'] == 1].index)
df_relevant = df_relevant.dropna(subset=['lastpage'])
df_relevant.shape


# In[7]:


# Create "Subject" pie chart

# Set plotting style for ICML 2022
plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=False))

# List most common subjects
# remove spaces behind subjects
df_relevant['Subject'] = df_relevant['Subject'].str.strip()
subject_counts = df_relevant['Subject'].value_counts()
print(subject_counts)

# Get top 3 subjects
top_subjects = subject_counts.nlargest(4)

# Calculate the sum of all other subjects
other_subjects = subject_counts.sum() - top_subjects.sum()

# Create a new DataFrame for the pie chart
pie_df = top_subjects
pie_df['Others'] = other_subjects

# Pie chart
# Create a pie chart
plt.figure(figsize=(10,6))
plt.pie(pie_df, labels = pie_df.index, autopct='%1.1f%%', startangle=140)

# Draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  

plt.show()


# ### Explore data with simple visualizations in the following

# In[9]:


# Get top 4 subjects
top_subjects = df_relevant['Subject'].value_counts().nlargest(4)

# Create a new DataFrame with top 4 subjects and 'Others'
df_relevant['TopSubjects'] = df_relevant['Subject'].where(df_relevant['Subject'].isin(top_subjects.index), 'Others')

# Create a list of colors
colors = cm.viridis(np.linspace(0, 1, 2))  # Use viridis colormap

# Create a crosstab DataFrame
cross_tab = pd.crosstab(df_relevant['TopSubjects'], df_relevant['GenAIUse'], normalize='index')*100

# Define custom x labels
custom_labels2 = ['Informatics \n (N=15)', 'Cognitive \n Science \n (N=10)', 'Machine \n Learning \n (N=8)', 
                  'Other \n (N=137)', 'Business \n Psychology \n (N=28)']

# Define custom legend labels
custom_labels_leg = ['Yes', 'No']

# Plot the DataFrame
ax = cross_tab.plot(kind='bar', stacked=True, color=colors)

#plt.title('Share of generative AI users by study subject')
plt.xlabel('Study subjects')
plt.ylabel('Percentage of study subject')
    
# Set the custom labels
ax.set_xticklabels(custom_labels2, rotation=0)

# Create a custom legend
handles, labels = ax.get_legend_handles_labels()
    
legend = ax.legend(handles, custom_labels_leg, loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1)

# Save the figure
plt.savefig('fig_genAIUseXSubject.pdf')

plt.show()


# In[ ]:




