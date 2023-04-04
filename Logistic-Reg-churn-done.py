#!/usr/bin/env python
# coding: utf-8

# ### Customer churn with Logistic Regression
# 
# A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out who is leaving and why.
# 

# In[1]:


get_ipython().system('pip install scikit-learn==0.23.1')


# Let's first import required libraries:
# 

# In[2]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# <h2 id="about_dataset">About the dataset</h2>
# We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company. 
# 
# The dataset includes information about:
# 
# *   Customers who left within the last month – the column is called Churn
# *   Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# *   Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# *   Demographic info about customers – gender, age range, and if they have partners and dependents
# 

# In[3]:


get_ipython().system('wget -O ChurnData.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv')


# ## Load Data From CSV File
# 

# In[4]:


churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()


# <h2 id="preprocessing">Data pre-processing and selection</h2>
# 

# Let's select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the skitlearn algorithm:
# 

# In[5]:


churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()


# How many rows and columns are in this dataset in total.

# In[6]:


churn_df.shape


# Let's define X, and y for our dataset:
# 

# In[7]:


X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[8]:


y = np.asarray(churn_df['churn'])
y [0:5]


# Also, we normalize the dataset:
# 

# In[9]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ## Train/Test dataset
# 

# We split our dataset into train and test set:
# With 80% train samples and 20% test sample

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <h2 id="modeling">Modeling (Logistic Regression with Scikit-learn)</h2>
# 

# Let's build our model using **LogisticRegression** from the Scikit-learn package. 

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# Now we can predict using our test set:
# 

# In[12]:


yhat = LR.predict(X_test)
yhat


# **predict_proba**  returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
# 

# In[13]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# <h2 id="evaluation">Evaluation</h2>
# 

# ### jaccard index
# 
# Let's try the jaccard index for accuracy evaluation.
# 

# In[19]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)


# ### confusion matrix
# 
# Another way of looking at the accuracy of the classifier is to look at **confusion matrix**.
# 

# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[16]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# Let's look at first row. The first row is for customers whose actual churn value in the test set is 1.
# As you can calculate, out of 40 customers, the churn value of 15 of them is 1.
# Out of these 15 cases, the classifier correctly predicted 6 of them as 1, and 9 of them as 0.
# 
# This means, for 6 customers, the actual churn value was 1 in test set and classifier also correctly predicted those as 1. However, while the actual label of 9 customers was 1, the classifier predicted those as 0, which is not very good. We can consider it as the error of the model for first row.
# 
# What about the customers with churn value 0? Lets look at the second row.
# It looks like  there were 25 customers whom their churn value were 0.
# 
# The classifier correctly predicted 24 of them as 0, and one of them wrongly as 1. So, it has done a good job in predicting the customers with churn value 0. A good thing about the confusion matrix is that it shows the model’s ability to correctly predict or separate the classes.  In a specific case of the binary classifier, such as this example,  we can interpret these numbers as the count of true positives, false positives, true negatives, and false negatives.
# 

# In[17]:


print (classification_report(y_test, yhat))


# Based on the count of each section, we can calculate precision and recall of each label.
# 
# 
# So, we can calculate the precision and recall of each class.
# 
# 
# 
# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.
# 

# ### log loss
# 
# Now, let's try **log loss** for evaluation. In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). This probability is a value between 0 and 1.
# Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
# 

# In[21]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# 
# Building Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ 

# In[22]:


LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))


# In[ ]:




