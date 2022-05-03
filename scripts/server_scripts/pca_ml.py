#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For K-means clustering

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA


# In[ ]:


# dirname = '/Users/shauntan2/Documents/Imperial College/Term 2/Machine Learning/ML_Project/data/Kather_texture_2016_image_tiles_5000/01_TUMOR'
folderpath = '../Kather_texture_2016_image_tiles_5000'
final = []
for i in os.listdir(folderpath):
    if not i.startswith('.'): # To ignore hidden files
        print(i)
        for fname in os.listdir(os.path.join(folderpath, i)):
            im = Image.open(os.path.join(folderpath, i, fname))
            imarray = np.array(im)
            final.append(imarray)

final = np.asarray(final)


# In[ ]:


# Reshape data for clustering
# Clustering of images require it to be 2D arrays
final_clustering = np.reshape(final, (final.shape[0], -1))
final_clustering.shape


# In[ ]:


# Data Normalisation
final_clustering_standard = StandardScaler().fit_transform(final_clustering)


# In[ ]:


# PCA before clustering
pca = PCA()
pca.fit(final_clustering_standard)


# In[ ]:

with open("pca_output.txt", "w") as external_file:
    print(pca.explained_variance_ratio_)
    external_file.close()


# In[ ]:


plt.figure(figsize = (10,8))
plt.plot(range(1,100), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestle = '--')
plt.title("Explained Variance by Components")
plt.xlabel('Number of Components')
plt.ylabel("Cumulative Explained Variance")
plt.savefig('pca_crc.pdf')


# In[ ]:





# In[ ]:




