#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[7]:


from datetime import datetime
import os 
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# For random forest

from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.inspection import permutation_importance


# In[5]:


start_time = datetime.now()


# # Data

# In[8]:


# dirname = '/Users/shauntan2/Documents/Imperial College/Term 2/Machine Learning/ML_Project/data/Kather_texture_2016_image_tiles_5000/01_TUMOR'
folderpath = '../Kather_texture_2016_image_tiles_5000'
final = []
img_labels = []
for i in os.listdir(folderpath):
    if not i.startswith('.'): # To ignore hidden files
        print(i)
        for fname in os.listdir(os.path.join(folderpath, i)):
            im = Image.open(os.path.join(folderpath, i, fname))
            imarray = np.array(im)
            final.append(imarray)
            img_labels.append(i)

final = np.asarray(final)
final


# In[24]:


len(img_labels)


# In[25]:


strings = {'02_STROMA': 2, '06_MUCOSA': 6, '05_DEBRIS': 5, '01_TUMOR': 1, '03_COMPLEX': 3, '08_EMPTY': 8, '04_LYMPHO': 4, '07_ADIPOSE': 7}
# Changing labels to integers
img_labels_integer = [strings[item] for item in img_labels]
img_labels_integer


# In[27]:


# Reshaping image arrays
final_forest = np.reshape(final, (final.shape[0], -1))
final_forest.shape


# In[30]:


# Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(final_forest, img_labels_integer, test_size=0.2, random_state=8)


# # Analysis

# In[35]:


# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)


# In[34]:


parameters = {'max_features':['sqrt'], 'n_estimators':[50,100,250],
              'max_depth':range(2,7),'min_samples_leaf':[2,4,6,8], 
              'criterion' :['gini', 'entropy']}

rf_class = GridSearchCV(RandomForestClassifier(random_state = 8),
                        parameters, n_jobs=3)
rf_class.fit(X = X_train, y = y_train)
rf_model = rf_class.best_estimator_

y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print('The best parameters are {}'.format(rf_class.best_params_))


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

