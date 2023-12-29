#!/usr/bin/env python
# coding: utf-8

# In[49]:


# First Stage: All the necessary package installations
import subprocess
subprocess.run(['pip', 'install', 'pandas', 'numpy', 'scikit-learn'])


# In[50]:


pip install scikit-learn


# In[51]:


# Second Stage: Loading the Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold


# In[52]:


# Third Stage: Load necessary data sets
#we have two dataset. 
#1. CSV File :- basic information about the songs along with the genre (Total 16460 different songs)
#2. CSV File :- containing muscial features like danceability , acousticness , valence etc

song_data = pd.read_csv(r"C:\Users\fma-rock-vs-hiphop.csv")
feature_data = pd.read_csv(r"C:\Users\echonest-metrics.csv")


# In[53]:


song_data


# In[54]:


feature_data


# In[55]:


# Fourth Stage: Data Exploration
print(song_data.head(10))
total_row_count = len(song_data)
print(total_row_count)
count_distinct_rows = len(song_data.drop_duplicates())
print(count_distinct_rows)
total_distinct_values = len(song_data['title'].unique())
print(total_distinct_values)

# Explore Feature_data
print(feature_data.info())
print(feature_data.describe())
print(feature_data.head(10))


# In[56]:


# Fifth Stage: Merge two data sets
merge_data = pd.merge(feature_data, song_data[['track_id', 'genre_top']], on='track_id')

# we will be using this merge data set through our analaysis. 


# In[57]:


merge_data


# In[58]:


# Sixth Stage: Explore merge data set

# 1. Find the correlation amoung the varaibles. The highely correlated variables should 
#    be removed and reduce the varaibles if possible. 

numeric_columns = merge_data.select_dtypes(include=[np.number]).columns
numeric_data = merge_data[numeric_columns]

correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()


# ### As no features exhibit a correlation of 0.8 or higher,
# ### I have determined that there are no highly correlated features. 
# ### Consequently, all variables or features will be considered for further analysis.

# In[ ]:


#### Seventh Stage: Normalize merge data set

# The next step is to normalize the data. our data can have various scale values present, 
# normalising the scale values helps preventing certain features from dominating others. 
# We will be using r scale function which uses Standardize features to normalise data. 

# First step would be creating two dataframes. 
predictors = merge_data.drop(['genre_top', 'track_id'], axis=1)
class_labels = merge_data['genre_top']

# Secound step would be Standardize the features using the scale function.
scaler = StandardScaler()
scaled_predictors = scaler.fit_transform(predictors)


# In[61]:


scaled_predictors


# ### We now have normalised values, but we still have one more issue that needs to be solved. 
# ### The issue is , we have too many features or varaibles which may lead to overfitting of model. 
# ### To reduce the variables or components we need to perform PCA (Principal component analysis)

# In[63]:


# Eighth Stage: Perform PCA (Principal component analysis) on Scaled_train_Predictors data

# We have two steps in this process. 
# First step, we will apply PCA to our Scaled_train_Predictors data and check if we are able to drop 
# any varaible. 

# Second step, if first steps fails then we will apply PCA on cummulative data and check if any varaible
# dropping is observed or not. 


# First step

# Step 1: Apply PCA
pca = PCA()
pca_outcome = pca.fit_transform(scaled_predictors)

# Step 2: Extract the explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Print the explained variance ratios
print("Explained variance ratio:", explained_variance)
print("Number of components =", len(explained_variance))

# Plot the explained variance using a Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Component #')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.show()


# In[64]:


# Second step
# Step 3: Calculate cumulative explained variance
cum_exp_variance = np.cumsum(explained_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cum_exp_variance) + 1), cum_exp_variance, marker='o', linestyle='-')
plt.axhline(y=0.9, linestyle='--', color='red')
plt.xlabel('Principal Component #')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()

# Step 4: Use only 6 components
n_components = 6
pca_result = PCA(n_components=n_components)
pca_projection = pca_result.fit_transform(scaled_predictors)

# Get the dimensions (shape) of pca_projection
shape = pca_projection.shape
print("Shape of pca_projection:", shape)


# ### Now that we have reduced the features to six, 
# ### we proceed to build models using two algorithms: decision tree and logistic regression.
# ### To initiate this process, the initial step involves splitting our data
# ### into training and testing sets, which will be represented by variables
# ### named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.

# In[67]:


# Ninth Stage: Split the data named Train_Predictors, Test_Predictors, Train_Class, and Test_Class
X_train, X_test, y_train, y_test = train_test_split(pca_projection, class_labels, test_size=0.3, random_state=10)


# In[68]:


# Tenth Stage: Modeling
# Here we have two models.
# A. Decision Tree
# B. Logistic Regression

# A. Decision Tree
dt_model = DecisionTreeClassifier(random_state=10)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Create Confusion Matrix and Statistics for decision tree model
conf_matrix_dt = confusion_matrix(y_test, dt_predictions)
accuracy_dt = accuracy_score(y_test, dt_predictions)

# Print the results
print("Decision Tree:")
print("Confusion Matrix:\n", conf_matrix_dt)
print("Accuracy:", accuracy_dt)


# In[69]:


# B. Logistic Regression
lg_model = LogisticRegression(random_state=10)
lg_model.fit(X_train, y_train)
lg_predictions = lg_model.predict(X_test)

# Create confusion matrix
conf_matrix_lg = confusion_matrix(y_test, lg_predictions)
accuracy_lg = accuracy_score(y_test, lg_predictions)

# Print the results
print("Logistic Regression:")
print("Confusion Matrix:\n", conf_matrix_lg)
print("Accuracy:", accuracy_lg)


# ##### By looking at both the model output it seems that the accuracy is almost same. 

# In[71]:


# Eleventh Stage: Checking and fixing balance dataset
# First step , check if data set is balanced or not.
# Second step, if not then fix it by sampling the data set.  

# First step 
class_counts = merge_data['genre_top'].value_counts()
print(class_counts)




# In[72]:


# Viewing the data- there is a class imbalance data. More Rock songs present in our
#dataset than Hip-Hop
#3892 rock songs compared to only 910 hip hop songs. 
#Remove this imbalance by sampling the rock songs to have the same number as hip hop


# In[73]:


# Second step: If not balanced, fix it by sampling the dataset
rock_data = merge_data[merge_data['genre_top'] == 'Rock']
hip_hop_data = merge_data[merge_data['genre_top'] == 'Hip-Hop']

# Sample rows from rock to match the number of rows in hip-hop
rock_data_sampled = resample(rock_data, replace=True, n_samples=len(hip_hop_data), random_state=10)

# Concatenate to create the balanced dataset
balanced_data = pd.concat([rock_data_sampled, hip_hop_data])


# In[74]:


balanced_data


# #### Now we have new dataset (balanced_data), perform all the steps from step 7 till step 10. 

# In[77]:


# Twelfth Stage: Perform all the steps from step 7 till step 10 for the balanced dataset
# 7. Normalise merge data set
# 8. Perform PCA
# 9. Split the data
# 10. Modelling (Decision Tree and Logistic Regression)



# Seventh Stage: Normalize merge data set

# The next step is to normalize the data. our data can have various scale values present, 
# normalising the scale values helps preventing certain features from dominating others. 
# We will be using r scale function which uses Standardize features to normalise data. 

# First step would be creating two dataframes. 
predictors = balanced_data.drop(['genre_top', 'track_id'], axis=1)
class_labels = balanced_data['genre_top']

# Secound step would be Standardize the features using the scale function.
scaler = StandardScaler()
scaled_predictors = scaler.fit_transform(predictors)



# Eighth Stage: Perform PCA (Principal component analysis) on Scaled_train_Predictors data

# We have two steps in this process. 
# First step, we will apply PCA to our Scaled_train_Predictors data and check if we are able to drop 
# any varaible. 

# Second step, if first steps fails then we will apply PCA on cummulative data and check if any varaible
# dropping is observed or not. 


# First step

# Step 1: Apply PCA
pca = PCA()
pca_outcome = pca.fit_transform(scaled_predictors)

# Step 2: Extract the explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Print the explained variance ratios
print("Explained variance ratio:", explained_variance)
print("Number of components =", len(explained_variance))

# Plot the explained variance using a Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Component #')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.show()


# Second step
# Step 3: Calculate cumulative explained variance
cum_exp_variance = np.cumsum(explained_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cum_exp_variance) + 1), cum_exp_variance, marker='o', linestyle='-')
plt.axhline(y=0.9, linestyle='--', color='red')
plt.xlabel('Principal Component #')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.show()

# Step 4: Use only 6 components
n_components = 6
pca_result = PCA(n_components=n_components)
pca_projection = pca_result.fit_transform(scaled_predictors)

# Get the dimensions (shape) of pca_projection
shape = pca_projection.shape
print("Shape of pca_projection:", shape)


# Now that we have reduced the features to six, 
# we proceed to build models using two algorithms: decision tree and logistic regression.
# To initiate this process, the initial step involves splitting our data
# into training and testing sets, which will be represented by variables
# named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.






# Ninth Stage: Split the data named Train_Predictors, Test_Predictors, Train_Class, and Test_Class
X_train, X_test, y_train, y_test = train_test_split(pca_projection, class_labels, test_size=0.3, random_state=10)




# Tenth Stage: Modeling
# Here we have two models.
# A. Decision Tree
# B. Logistic Regression

# A. Decision Tree
dt_model = DecisionTreeClassifier(random_state=10)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Create Confusion Matrix and Statistics for decision tree model
conf_matrix_dt = confusion_matrix(y_test, dt_predictions)
accuracy_dt = accuracy_score(y_test, dt_predictions)

# Print the results
print("Decision Tree:")
print("Confusion Matrix:\n", conf_matrix_dt)
print("Accuracy:", accuracy_dt)



# B. Logistic Regression
lg_model = LogisticRegression(random_state=10)
lg_model.fit(X_train, y_train)
lg_predictions = lg_model.predict(X_test)

# Create confusion matrix
conf_matrix_lg = confusion_matrix(y_test, lg_predictions)
accuracy_lg = accuracy_score(y_test, lg_predictions)

# Print the results
print("Logistic Regression:")
print("Confusion Matrix:\n", conf_matrix_lg)
print("Accuracy:", accuracy_lg)



# ### Now the prediction of hip-hop bad rock songs seems to be fine. 

# In[87]:


#  Thirteen Stage: Validation using cross-validation

cv_dt = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_scores_dt = cross_val_score(dt_model, pca_projection, class_labels, scoring='accuracy', cv=cv_dt, n_jobs=-1)

cv_lg = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_scores_lg = cross_val_score(lg_model, pca_projection, class_labels, scoring='accuracy', cv=cv_lg, n_jobs=-1)

print("Decision Tree Classifier Cross-Validation Accuracy:", np.mean(cv_scores_dt))
print("Logistic Regression Cross-Validation Accuracy:", np.mean(cv_scores_lg))


# ### Hence Logistic Regression has better accuracy than decision tree. Therefore Logistic Regression model is used 
# ### to classify-song-genres-from-audio-data

# In[ ]:




