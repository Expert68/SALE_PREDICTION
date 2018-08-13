# 特征工程及数据清洗
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin


# ---------------------------定义数据填充类-------------------------
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# ---------------------------定义特征工程类-------------------------
class DataProcess:

    def __init__(self, train_df, test_df, features, features_non_numeric):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.features_non_numeric = features_non_numeric

    def process_data(self):
        for data in [self.train_df, self.test_df]:
            data['year'] = data.Date.apply(lambda x: x.split('-')[0])
            data['year'] = data['year'].astype(float)
            data['month'] = data.Date.apply(lambda x: x.split('-')[1])
            data['month'] = data['month'].astype(float)
            data['day'] = data.Date.apply(lambda x: x.split('-')[2])
            data['day'] = data['day'].astype(float)
            data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
            data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
            data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
            data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
            data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
            data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
            data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
            data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
            data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
            data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
            data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
            data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

        self.noise_features = ['Id', 'Date', 'PromoInterval']
        self.features = [c for c in self.features if c not in self.noise_features]
        self.features_non_numeric = [c for c in self.features_non_numeric if c not in self.noise_features]
        self.features.extend(['year', 'month', 'day'])

    def fillna_(self):
        datafiller = DataFrameImputer()
        self.train_df = datafiller.fit_transform(self.train_df)
        self.test_df = datafiller.fit_transform(self.test_df)

    def encoder(self):
        le = LabelEncoder()
        for col in self.features_non_numeric:
            self.train_df[col] = le.fit_transform(self.train_df[col])
            self.test_df[col] = le.fit_transform(self.test_df[col])
        self.all_df = pd.concat((self.train_df,self.test_df),sort=False)
        return self.all_df

    def std_scaler(self):
        scaler = StandardScaler()
        for col in (set(self.features) - set(self.features_non_numeric)-set(self.noise_features)):
            scaler.fit(self.all_df[col].values.reshape(-1,1))
            self.train_df[col] = scaler.transform(self.train_df[col].values.reshape(-1,1))
            self.test_df[col] = scaler.transform(self.test_df[col].values.reshape(-1,1))
        return self.train_df, self.test_df






