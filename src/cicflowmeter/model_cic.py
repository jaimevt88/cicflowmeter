from unicodedata import category
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


import pickle  # To load data int disk


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class ClasifierRf:

    def __init__(self):

            self.filename = '/home/sdnonos/cicflowmeter/src/cicflowmeter/finalized_model.sav'
            self.loaded_model = pickle.load(open(self.filename, 'rb'))

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
        #save_model_rf()
        

    def clean_df(self, data):
        df_cl = pd.DataFrame.from_dict([data])
        df_cl.columns=df_cl.columns.str.strip().str.lower()
        df_cl.columns = df_cl.columns.str.replace('_',' ')
        df_cl = df_cl.drop(['fwd urg flags', 'bwd urg flags','fwd psh flags', 'bwd psh flags','fwd byts b avg', 
        'fwd pkts b avg', 'fwd byts b avg','bwd byts b avg', 'bwd pkts b avg',
            'fwd blk rate avg', 'bwd blk rate avg', 'bwd blk rate avg','cwe flag count','timestamp'], axis=1)
        df_cl.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_cl.dropna(inplace=True)
        df_cl.reset_index(drop=True)
        return df_cl

    def save_model_rf(self, filename):
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model_rf(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))

    def predict_rf(self, data):
        #print(data[self.sel_feat])
        y_predicted = self.loaded_model.predict(data[self.sel_feat])
        return y_predicted




