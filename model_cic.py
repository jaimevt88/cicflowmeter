import numpy as np
import pandas as pd
import statsmodels.api as sm

#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale

import pickle  # To load data int disk
from prettytable import PrettyTable  # To print in tabular format

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict


class ClasifierRf:

    def __init__(self):
            self.sel_feat = ['tot bwd pkts', 'bwd iat min', 'subflow bwd pkts',
            'flow iat std', 'fwd pkt len min', 'bwd header len',
            'flow iat min', 'bwd iat std', 'init fwd win byts', 'active std',
            'urg flag cnt', 'init bwd win byts', 'dst port',
            'pkt len mean', 'bwd pkt len std']

    def train_rf(self):
        df_attack=pd.read_csv("/home/sdnonos/flows.csv")
        df_cl = clean_df(df_attack)
        att = {'BENIGN':0, 'DoS slowloris':1, 'DoS Slowhttptest':1, 'DoS Hulk':1, 'DoS GoldenEye':1, 'Heartbleed':1}
        df_cl = df_cl.replace({'label':att},inplace=True)
        X = df_cl.drop(['label'],axis=1)
        y = df_cl['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
        model = RandomForestClassifier(n_estimators=5)
        model.fit(X_train[self.sel_feat], y_train)
        save_model_rf()
        

    def clean_df(df_cl):
        df_cl.columns=df_cl.columns.str.strip().str.lower()
        df_cl.columns = df_cl.columns.str.replace('_',' ')
        df_cl = df_cl.drop(['fwd urg flags', 'bwd urg flags','fwd psh flags', 'bwd psh flags','fwd byts b avg', 
        'fwd pkts b avg', 'fwd byts b avg','bwd byts b avg', 'bwd pkts b avg',
            'fwd blk rate avg', 'bwd blk rate avg', 'bwd blk rate avg','cwe flag count','timestamp'], axis=1)
        df_cl.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_cl.dropna(inplace=True)
        df_cl.reset_index(drop=True)
        return df_cl

    def save_model_rf(filename):
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model_rf(filename):
        loaded_model = pickle.load(open(filename, 'rb'))

    def predict_rf(data):
        y_predicted = loaded_model.predict(data[self.sel_feat])
        return y_predicted




