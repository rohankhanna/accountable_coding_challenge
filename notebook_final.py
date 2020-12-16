#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tabloo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('concatenated.csv')


# In[3]:


df


# In[4]:


for column in df.columns:
    print(column, df[column][df[column].isnull().values].shape)


# In[5]:


df[['date_scraped','mois']]


# In[6]:


df['commune'].unique()


# In[7]:


df[['commune_str','postal_code']] = df['commune'].str.split('/',expand=True)


# In[8]:


df


# In[9]:


columns_to_remove = ['mois', 'DE_commune', 'FR_commune','NL_commune', 'valid?', 'generated', 'code_postal', 'province', 'gadm_commune', 'immoweb_commune', 'url', 'date_scraped', 'Unnamed: 0']
for column in columns_to_remove:
    del df[column]


# In[10]:


df


# In[11]:


# since commune has been split into 2 columns, we can remove commune
del df['commune']


# In[12]:


df


# In[13]:


df.dropna()
# this seems to be pretty clean data. I think this data can be modellable via some serial combination of decision tree, random forest, and/or regression


# In[14]:


data = df[ ['prix','surface_habitable','prix_par_mc'] ]


# In[15]:


df.describe()


# In[16]:


plt.figure(figsize=(3, 3))
sns.heatmap(data.corr().abs(),  annot=True)


# In[17]:


for k, v in data.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


# #TODO remove

# print(__doc__)

# # Import the necessary modules and libraries
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt

# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))

# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_3 = DecisionTreeRegressor(max_depth=10)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# regr_3.fit(X, y)

# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
# y_3 = regr_3.predict(X_test)

# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.plot(X_test, y_3, color="red", label="max_depth=10", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


# In[ ]:





# In[19]:


# TODO convert categorical variables to one-hot-encodings
categorical_variables = ['commune_str', 'postal_code']
commune_str_columns = list(pd.get_dummies(df['commune_str'], prefix='commune_str').columns)
df[commune_str_columns] = pd.get_dummies(df['commune_str'], prefix='commune_str')
postal_code_columns = list(pd.get_dummies(df['postal_code'], prefix='postal_code').columns)
df[postal_code_columns] = pd.get_dummies(df['postal_code'], prefix='postal_code')


# In[20]:


del df['commune_str']
del df['postal_code']


# In[21]:


df


# In[ ]:





# In[ ]:


X = df.loc[:, df.columns != 'prix'].to_numpy()
y = df['prix'].to_numpy()
with open('X.npy', 'wb') as f:
    np.save(f, X)
with open('y.npy', 'wb') as f:
    np.save(f, y)

# # In[ ]:


# X.shape


# # In[ ]:


# y.shape


# # In[ ]:


# print(__doc__)

# # Import the necessary modules and libraries
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# np.random.seed(120)

# # Create a random dataset
# rng = np.random.RandomState(1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_3 = DecisionTreeRegressor(max_depth=10)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
# regr_3.fit(X, y)

# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
# y_3 = regr_3.predict(X_test)

# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.plot(X_test, y_3, color="red", label="max_depth=10", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


# # In[25]:




# In[ ]:




