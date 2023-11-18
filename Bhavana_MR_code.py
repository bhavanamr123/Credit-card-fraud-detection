# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:25:40 2023

@author: bhava
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


data_train = pd.read_csv("C:\\Users\\bhava\\Desktop\\7th sem\\ML\\project phase-2\\training_data.csv")
classlabels = pd.read_csv("C:\\Users\\bhava\\Desktop\\7th sem\\ML\\project phase-2\\train_data_classlabels.csv")
data_test= pd.read_csv("C:\\Users\\bhava\\Desktop\\7th sem\\ML\\project phase-2\\testing_data.csv")



data = pd.concat([data_train, classlabels], axis=1)

data = shuffle(data, random_state=42)

fraud = data[data['Class']==1]
normal=data[data['Class']==0]

if data.isnull().values.any():
    print("There are missing values in the DataFrame.")
else:
    print("There are no missing values in the DataFrame.")

#cheching percentage distrbtn of fraud and normal
classes=data['Class'].value_counts()
normal_share=classes[0]/data['Class'].count()*100
fraud_share=classes[1]/data['Class'].count()*100

print('The percentage distribution of normal classes is:', normal_share)
print('The percentage distribution of fraud classes is:', fraud_share)

normal_stats = normal.describe()
fraud_stats = fraud.describe()
print("Statistics for Normal Transactions:")
print(normal_stats)
print("\nStatistics for Fraud Transactions:")
print(fraud_stats)

print('The plot showing the overview of the data: ')
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i] , ha = 'center' , fontsize = 15)

plt.figure(figsize=(12,10))
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xlabel('Class', fontsize = 20)
plt.ylabel('Frequency' , fontsize = 20)
plt.title('Transactions',fontsize = 30)
x = ['Normal','Fraud']
y = [56974 , 142]
addlabels(x, y)
plt.show()

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(method='pearson'), cmap= None)
plt.show()


X = data.drop('Class', axis=1)
y = data['Class']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=100, test_size=0.20, stratify=y)


# Classifiers
classifiers = {
    'Logistic Regression': (LogisticRegression(solver='liblinear', class_weight='balanced'), {'random_state':(0,10)}),
    
    'SVM': (SVC(class_weight='balanced', probability=True), {'C': [0.1, 1, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}),
    
    'Decision Tree': (DecisionTreeClassifier(random_state=40), {'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 40, 45, 60], 'ccp_alpha': [0.009, 0.01, 0.05, 0.1]}),
    
    'Random Forest': (RandomForestClassifier(max_features=None, class_weight='balanced'), {'criterion': ['entropy', 'gini'], 'n_estimators': [30, 50, 100], 'max_depth': [10, 20, 30, 50, 100, 200]}),
}

best_model_name = None
best_model = None
best_predictions = None
best_accuracy = 0.0

classifier_names = list(classifiers.keys())  # Extract classifier names

for i, ((classifier_name, (classifier, param_grid)), k) in enumerate(zip(classifiers.items(), range(5, 20))):
    selector = SelectKBest(chi2, k=k)
    X_chi2 = selector.fit_transform(X_train, y_train)
    selected_feature_indices = selector.get_support(indices=True)
    
    selected_feature_names = X.columns[selected_feature_indices]
    
    # Use the counter to select the appropriate classifier name
    current_classifier_name = classifier_names[i]
    
    print(f"Selected Features for {current_classifier_name}:")
    print(selected_feature_names)
    
    X_test_chi2 = selector.transform(X_test)
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_chi2, y_train)
    
    best_params = grid_search.best_params_
    current_best_model = grid_search.best_estimator_
    y_pred = current_best_model.predict(X_test_chi2)
    
    # Evaluation
    current_accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", current_accuracy)
    print("\n")

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model_name = current_classifier_name
        best_model = current_best_model
        best_predictions = y_pred

print(f"Best Model: {best_model_name}")
print("Best Parameters:", best_params)
print("Predicted Class Labels:")
print(best_predictions)
print("Actual Class Labels:")
print(y_test.values)



output_df = pd.DataFrame({'Actual_Class': y_test.values, 'Predicted_Class': best_predictions})
output_df.to_csv('predictions_comparison.csv', index=False)