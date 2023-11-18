# Credit-card-fraud-detection
With technological developments, credit cards have become an integral part of our day-to-day life. A
credit card comes with an option to buy anything without the constraints of paying the full amount
at the moment. With this popularity of credit cards comes the other side, which is fraudulent credit
card transactions. They involve stealing one’s credit card information and making unauthorised trans-
actions. The high frequency of such cases invokes the urgency to find a suitable mechanism to detect
such fraudulent transactions. The objective of this project is to find the best model that can detect
fraudulent transactions from a given dataset. The dataset contains features V1, V2, . . . V28, which
are PCA transformed; the features that are not transformed are time and amount. Feature ’Time’
contains the seconds elapsed between each transaction and the first transaction. There are 57116
entries in the training set, including 30 columns. The test set consists of 14280 entries. There are no
null values in the dataset. The ’Class’ is the response variable, which takes value 1 in case of fraud and
0 otherwise. The data is highly unbalanced, as there are only 142 fraudulent transactions.(Figure 1).
Since this is a classification problem, logistic regression is primarily performed. Other machine learn-
ing models such as SVM, Decision tree algorithm and Random Forest Classifier are also performed.
This is followed by a model evaluation in order to find out which is the best-performing model. The
results show that the Random Forest algorithm is the best-performing model with an accuracy score
of 99.89 per cent out of the rest three models
