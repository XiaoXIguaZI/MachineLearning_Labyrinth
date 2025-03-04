import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# getdata
riskdat =  pd.read_csv('数据集.csv')
riskdat_c = riskdat.copy()

# dim: 31713 samples, 59 features, 1 label
print(riskdat.shape)
print(riskdat.head())
# + - distribution
print(riskdat.hot.value_counts())
# missing variables
riskdat.missing_var = riskdat.isnull().sum(axis=1)
riskdat=riskdat.loc[riskdat.missing_var < 10,:]
print(riskdat.shape)
# delete all-zero variables
riskdat=riskdat.loc[:,~(riskdat == 0).all(axis=0)]
riskdat.shape
# delete non-data variables
riskdat_sub2 = riskdat.select_dtypes(include=['object'])
riskdat_sub2.head(3)
riskdat_sub1=riskdat.select_dtypes(exclude=['object'])
riskdat_sub1.head(3)
riskdat_sub1 = riskdat_sub1.fillna(riskdat_sub1.mean())
print(riskdat_sub1.shape)

# define X and y
X1, y1 = riskdat_sub1.iloc[:, 0:-1], riskdat_sub1.hot
print(X1.shape, y1.shape)


# import libraries for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import seaborn as sns

# slpit training set and test set
X1_train, X1_val, y1_train, y1_val = train_test_split(X1.values, y1.values, test_size=0.3, random_state=114)
print(X1_train.shape, y1_train.shape, X1_val.shape, y1_val.shape)

# data discription
DD = riskdat_sub1.describe()
DD






# %%
# class balance (positive and negetive) _ for 1^ LOGISTIC REGRESSION
class_balance = y1.value_counts()

# Bar chart by counts
plt.figure(figsize=(8, 6))
class_balance.plot(kind='bar', color=['lightblue', 'lavender'])
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Balance - Bar Chart')
plt.xticks(rotation=0)
plt.show()

# Pie chart by proportion
plt.figure(figsize=(8, 8))
class_balance.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lavender'])
plt.title('Class Balance - Pie Chart')
plt.ylabel('')
plt.show()



# %%
# 1^ LOGISTIC REGRESSION

# 1.1^ default parms
lr_clf1 = LogisticRegression(class_weight={0: 0.47, 1: 0.53})
lr_clf1.fit(X1_train, y1_train)
y1_train_pred = lr_clf1.predict(X1_train)
print("Confusion matrix (training):\n {0}\n".format(confusion_matrix(y1_train, y1_train_pred)))
print("Classification report (training):\n {0}".format(classification_report(y1_train, y1_train_pred)))

y1_val_pred = lr_clf1.predict(X1_val)
print("Confusion matrix (validation):\n {0}\n".format(confusion_matrix(y1_val, y1_val_pred)))
print("Classification report (validation):\n {0}".format(classification_report(y1_val, y1_val_pred)))

# prams tuning
lr_clf_tuned = LogisticRegression(class_weight={0: 0.47, 1: 0.53})
lr_clf_params = {
   "penalty": ["l1", "l2"],
    "C": [1, 1.3, 1.5, 2]
}
lr_clf_cv = GridSearchCV(lr_clf_tuned, lr_clf_params, cv=5)
lr_clf_cv.fit(X1_train, y1_train)
print(lr_clf_cv.best_params_)

# 1.2^ optimal parms
lr_clf2 = LogisticRegression(penalty="l2", C=2, class_weight={0: 0.47, 1: 0.53})
lr_clf2.fit(X1_train, y1_train)
y1_train_pred = lr_clf2.predict(X1_train)
print("Confusion matrix (training):\n {0}\n".format(confusion_matrix(y1_train, y1_train_pred)))
print("Classification report (training):\n {0}".format(classification_report(y1_train, y1_train_pred)))

y1_val_pred = lr_clf2.predict(X1_val)
print("Confusion matrix (validation):\n {0}\n".format(confusion_matrix(y1_val, y1_val_pred)))
print("Classification report (validation):\n {0}".format(classification_report(y1_val, y1_val_pred)))

# ROC
y1_valid_score_lr1 = lr_clf1.predict_proba(X1_val)
y1_valid_score_lr2 = lr_clf2.predict_proba(X1_val)

fpr_lr1, tpr_lr1, thresholds_lr1 = roc_curve(y1_val, y1_valid_score_lr1[:, 1])
fpr_lr2, tpr_lr2, thresholds_lr2 = roc_curve(y1_val, y1_valid_score_lr2[:, 1])

roc_auc_lr1 = auc(fpr_lr1, tpr_lr1)
roc_auc_lr2 = auc(fpr_lr2, tpr_lr2)

plt.plot(fpr_lr1, tpr_lr1, fpr_lr2, tpr_lr2, lw=2, alpha=.6)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.legend(["Logistic Reg 1 (AUC {:.4f})".format(roc_auc_lr1),
            "Logistic Reg 2 (AUC {:.4f})".format(roc_auc_lr2)], fontsize=8, loc=2)

plt.show()






# %%
# 2^ Disision Tree
dt_clf = DecisionTreeClassifier(class_weight={0: 0.47, 1: 0.53}, random_state=123)
dt_clf.fit(X1_train, y1_train)

y1_train_pred_dt = dt_clf.predict(X1_train)
print("Confusion matrix (training - Decision Tree):\n", confusion_matrix(y1_train, y1_train_pred_dt))
print("Classification report (training - Decision Tree):\n", classification_report(y1_train, y1_train_pred_dt))

y1_val_pred_dt = dt_clf.predict(X1_val)
print("Confusion matrix (validation - Decision Tree):\n", confusion_matrix(y1_val, y1_val_pred_dt))
print("Classification report (validation - Decision Tree):\n", classification_report(y1_val, y1_val_pred_dt))

y1_valid_score_dt = dt_clf.predict_proba(X1_val)

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y1_val, y1_valid_score_dt[:, 1])
roc_auc_dt = auc(fpr_dt, tpr_dt)

plt.plot(fpr_lr2, tpr_lr2, fpr_dt, tpr_dt, lw=2, alpha=.6)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve - Logistic Reg 2 vs Decision Tree")
plt.legend(["Logistic Reg 2 (AUC {:.4f})".format(roc_auc_lr2),
            "Decision Tree (AUC {:.4f})".format(roc_auc_dt)], fontsize=8, loc=2)

plt.show()






# %%
# parms tunning (Bayesian) _ for 3^ RANDOM FOREST
def optimize_rf(class_weight_0, class_weight_1, random_state):
    class_weight = {0: class_weight_0, 1: class_weight_1}
    rf_clf = RandomForestClassifier(class_weight=class_weight, random_state=int(random_state))
    rf_clf.fit(X1_train, y1_train)
    y_val_pred = rf_clf.predict(X1_val)
    score = -f1_score(y1_val, y_val_pred, average='weighted')
    return score

pbounds = {'class_weight_0': (0.4, 0.6),
           'class_weight_1': (0.4, 0.6),
           'random_state': (0, 1000)}

optimizer = BayesianOptimization(
    f=optimize_rf,
    pbounds=pbounds,
    random_state=123,
)
optimizer.maximize(init_points=5, n_iter=10)
best_params = optimizer.max['params']
print(best_params)



# %%
# 3^ RANDOM FOREST
rf_clf = RandomForestClassifier(class_weight={0: 0.45892, 1: 0.55449}, random_state=480)
rf_clf.fit(X1_train, y1_train)

y1_train_pred_rf = rf_clf.predict(X1_train)
print("Confusion matrix (training - Random Forest):\n", confusion_matrix(y1_train, y1_train_pred_rf))
print("Classification report (training - Random Forest):\n", classification_report(y1_train, y1_train_pred_rf))

y1_val_pred_rf = rf_clf.predict(X1_val)
print("Confusion matrix (validation - Random Forest):\n", confusion_matrix(y1_val, y1_val_pred_rf))
print("Classification report (validation - Random Forest):\n", classification_report(y1_val, y1_val_pred_rf))

y1_valid_score_rf = rf_clf.predict_proba(X1_val)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y1_val, y1_valid_score_rf[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_lr2, tpr_lr2, fpr_rf, tpr_rf, lw=2, alpha=.6)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve - Logistic Reg 2 vs Random Forest")
plt.legend(["Logistic Reg 2 (AUC {:.4f})".format(roc_auc_lr2),
            "Random Forest (AUC {:.4f})".format(roc_auc_rf)], fontsize=8, loc=2)

plt.show()






# %%
# 4^ RANDOM FOREST wiht PCA
pca = PCA(n_components=25)
X1_train_pca = pca.fit_transform(X1_train)
X1_val_pca = pca.transform(X1_val)

rf_clf_pca = RandomForestClassifier(class_weight={0: 0.45892, 1: 0.55449}, random_state=480)
rf_clf_pca.fit(X1_train_pca, y1_train)

y1_train_pred_pca = rf_clf_pca.predict(X1_train_pca)
print("Confusion matrix (training - Random Forest):\n", confusion_matrix(y1_train, y1_train_pred_pca))
print("Classification report (training - Random Forest):\n", classification_report(y1_train, y1_train_pred_pca))

y1_val_pred_pca = rf_clf_pca.predict(X1_val_pca)
print("Confusion matrix (validation - Random Forest):\n", confusion_matrix(y1_val, y1_val_pred_pca))
print("Classification report (validation - Random Forest):\n", classification_report(y1_val, y1_val_pred_pca))

y1_valid_score_pca = rf_clf_pca.predict_proba(X1_val_pca)

fpr_pca, tpr_pca, thresholds_pca = roc_curve(y1_val, y1_valid_score_pca[:, 1])
roc_auc_pca = auc(fpr_pca, tpr_pca)

plt.plot(fpr_rf, tpr_rf, fpr_pca, tpr_pca, lw=2, alpha=.6)
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve - Random Forest 2 vs PCA")
plt.legend(["Random Forest (AUC {:.4f})".format(roc_auc_rf),
            "PCA (AUC {:.4f})".format(roc_auc_pca)], fontsize=8, loc=2)

plt.show()



# %%
# choose features _ for 4^ RANDOM FOREST wiht PCA
pca = PCA()
pca.fit(X1_train)

# accum explained variance curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 
         marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

# correlation heatmap
corr_matrix = X1.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of 59 Vectors')
plt.show()


