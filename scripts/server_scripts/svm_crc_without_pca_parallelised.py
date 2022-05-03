#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[6]:


from datetime import datetime
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

# SVM
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[ ]:


start_time = datetime.now()


# # Data

# In[2]:


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


# In[3]:


strings = {'02_STROMA': 2, '06_MUCOSA': 6, '05_DEBRIS': 5, '01_TUMOR': 1, '03_COMPLEX': 3, '08_EMPTY': 8, '04_LYMPHO': 4, '07_ADIPOSE': 7}
# Changing labels to integers
img_labels_integer = [strings[item] for item in img_labels]
img_labels_integer


# In[4]:


# Reshaping image arrays
final_svm = np.reshape(final, (final.shape[0], -1))
final_svm.shape


# In[7]:


# Data Normalisation
final_svm_standard = StandardScaler().fit_transform(final_svm)


# In[ ]:


# pca = PCA(n_components = 0.80)
# pca.fit(final_svm_standard)
# reduced = pca.transform(final_svm_standard)


# In[8]:


# Training and Test Data
X_train, X_test, y_train, y_test = train_test_split(final_svm_standard, img_labels_integer, test_size=0.2, random_state=8)


# # Analysis

# In[9]:


param_grid = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1, 'scale', 'auto'], 'kernel':['linear','rbf']}
svc = svm.SVC(probability = False)
model = GridSearchCV(svc, param_grid = param_grid, n_jobs = 6)


# In[10]:


# Model Training
model.fit(X_train, y_train)
svm_model_best = model.best_estimator_
print("The model is trained")


# In[ ]:


# Model Testing
y_pred = svm_model_best.predict(X_test)


# In[ ]:


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = "macro")
precision = precision_score(y_test, y_pred, average = "macro")
recall = recall_score(y_test, y_pred, average = "macro")

with open("svm_no_pca_parallelised_tuning.txt", "w") as external_file:
    print('accuracy = {}, f1 = {}. precision = {}, recall = {}'.format(acc, f1, precision, recall), file = external_file)
    print('The best parameters are {}'.format(model.best_params_), file = external_file)
    external_file.close()


# In[ ]:


# Confusion Matrix
labels = [1, 2, 3, 4, 5, 6, 7, 8]
cm = confusion_matrix(y_test, y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['tumor', 'stroma', 'complex', 'lympho', 'debris', 'mucosa', 'adipose', 'empty']) 
ax.yaxis.set_ticklabels(['tumor', 'stroma', 'complex', 'lympho', 'debris', 'mucosa', 'adipose', 'empty'])
ax.tick_params(axis='both', which='major', labelsize= 7)

plt.show()
plt.savefig('svm_confusion_crc_tuning.pdf')


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

