import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Metrics
from sklearn.metrics import f1_score, recall_score
# Model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# Fine-tune
from sklearn.model_selection import RandomizedSearchCV

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.drop(['Class'], axis=1)
y_train = df_train['Class']
X_test = df_test.drop(['Class'], axis=1)
y_test = df_test['Class']

label_weight_perc = df_train['Class'].value_counts(normalize=True) * 100
print(label_weight_perc)

#outliers
sns.boxplot(data= df_train, y='Amount')
plt.show()

feature_outlier = dict()

class Outlier:
  def __init__(self, q1, q3):
    self.q1 = q1
    self.q3 = q3
    self.iqr = q3 - q1
  def get_outlier_boundary(self):
    lower_fence = self.q1 - 1.5 * self.iqr
    upper_fence = self.q3 + 1.5 * self.iqr

    return lower_fence, upper_fence


def filter_outlier(df, cols=[]):
  if 'is_outlier' not in df.columns:
    df['is_outlier'] = (False) * len(df)
  for col in cols:
    if col in feature_outlier.keys():
      outlier = feature_outlier[col]
    else:
      q1 = df[col].quantile(0.25)
      q3 = df[col].quantile(0.75)
      outlier = Outlier(q1, q3)
      feature_outlier[col] = outlier
    lower_fence, upper_fence = outlier.get_outlier_boundary()
    outlier = (df[col] < lower_fence) | (df[col] > upper_fence)
    df['is_outlier'] = outlier | df['is_outlier']
df = df[~df['is_outlier']]
df = df.drop(['is_outlier'], axis=1)
return df

train_df = filter_outlier(df_train, cols=['Amount'])

#feature scaling
std_feat = ['Amount', 'Time']
std_pipeline = Pipeline([
('std_scaler', StandardScaler())])

#transformation
full_pipeline = ColumnTransformer([('std_feat', std_pipeline, std_feat)], remainder='passthrough')

df_val = pd.read_csv('val.csv')
X_val = df_test.drop(['Class'], axis=1)
y_val = df_test['Class']

X_train = train_df.drop(['Class'], axis=1)
y_train = train_df['Class']
X_train = full_pipeline.fit_transform(X_train)

X_val = df_val.drop(['Class'], axis=1)
y_val = df_val['Class']
X_val = full_pipeline.transform(X_val)

#model evaluation
model_eval = {
'model': [],
'recall': [],
'f1_score': []
}

def add_model_eval(model, recall, f1_score):
  model_eval['model'].append(model)
  model_eval['recall'].append(f'{recall: .2f}')
  model_eval['f1_score'].append(f'{f1_score: .2f}')

def view_models_eval(sort=False):
  eval_df = pd.DataFrame(model_eval)
  if sort:
    eval_df = eval_df.sort_values(by=['recall', 'f1_score'], ascending=[False, False])

  display(eval_df.style.hide_index())

log_reg = LogisticRegression(random_state=42, verbose=1)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_val)
add_model_eval('logistic regression', recall_score(y_val, y_pred), f1_score(y_val, y_pred))

view_models_eval()

sgd_clf = SGDClassifier(random_state=42, verbose=1)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_val)
add_model_eval('sgd classifier', recall_score(y_val, y_pred), f1_score(y_val, y_pred))

forest_clf = RandomForestClassifier(random_state=42, verbose=2, n_jobs=4)
forest_clf.fit(X_train, y_train)

y_pred = forest_clf.predict(X_val)
add_model_eval('random forest classifier', recall_score(y_val, y_pred), f1_score(y_val, y_pred))

view_models_eval()
