import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_matrix = pd.read_csv('./data_matrix.csv', index_col=0)
classes = pd.read_csv('./classes.csv', index_col=0)['class']

# Ensure data is loaded correctly
print(data_matrix.head())
print(classes.head())

# Task 1: Calculate the standardized data matrix
mean = data_matrix.mean(axis=0)
sttd = data_matrix.std(axis=0)
data_matrix_standardized = (data_matrix - mean) / sttd 

# Task 2: Perform PCA by fitting and transforming the data matrix
pca = PCA()
principal_components = pca.fit_transform(data_matrix_standardized)
print(f'Number of features in the principal components: {principal_components.shape[1]}')
print(f'Number of features in the original data matrix: {data_matrix.shape[1]}')

# Task 3: Calculate the eigenvalues from the singular values and extract the eigenvectors
singular_values = pca.singular_values_
eigenvalues = singular_values ** 2
eigenvectors = pca.components_.T

# Task 4: Extract the variance ratios
principal_axes_variance_ratios = pca.explained_variance_ratio_
principal_axes_variance_percents = principal_axes_variance_ratios * 100

# Task 5: Perform PCA once again but with 2 components
pca = PCA(n_components=2) 
principal_components = pca.fit_transform(data_matrix_standardized)
print(f'Number of Principal Components Features: {principal_components.shape[1]}')
print(f'Number of Original Data Features: {data_matrix_standardized.shape[1]}')

# Task 6: Plot the principal components and have its class as its hue
principal_components_data = pd.DataFrame({
     'PC1': principal_components[:, 0],
     'PC2': principal_components[:, 1],
     'class': classes,
 })

sns.lmplot(x='PC1', y='PC2', data=principal_components_data, hue='class', fit_reg=False)
plt.show()

# Convert class labels to numeric codes
y = classes.astype('category').cat.codes

# Task 7: Fit the transformed features onto the classifier and generate a score
pca_1 = PCA(n_components=2)
X = pca_1.fit_transform(data_matrix_standardized)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
svc_1 = LinearSVC(random_state=0, tol=1e-5)
svc_1.fit(X_train, y_train)
score_1 = svc_1.score(X_test, y_test)
print(f'Score for model with 2 PCA features: {score_1}')

# Task 8: Fit the classifier with the first two features of the original data matrix and generate a score
first_two_original_features = [0, 1]
X_original = data_matrix_standardized.iloc[:, first_two_original_features]
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.33, random_state=42)
svc_2 = LinearSVC(random_state=0, tol=1e-5)
svc_2.fit(X_train, y_train)
score_2 = svc_2.score(X_test, y_test)
print(f'Score for model with 2 original features: {score_2}')
