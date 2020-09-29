from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import seaborn as sns


class DataMangler:
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.stats = {}
        self.categorical_columns = []
        self.categorical_dims = {}

    def drop(self, col):
        if not isinstance(col, list):
            col = [col]

        try:
            for c in col:
                self.data = self.data.drop(c, axis=1)
                del self.stats[c]
        except (IndexError, KeyError) as e:
            print(f'{e}: Invalid index or key')

    def eda(self):
        print(f"Data:\n\
                Shape: \t{self.data.shape}\n\
                ")
        for col in self.data.columns:
            self.stats[col] = [self.data[col].nunique(),
                               self.data[col].unique()[:10],
                               self.data[col].min() if self.data[col].dtype != 'object' else '-',
                               self.data[col].max() if self.data[col].dtype != 'object' else '-',
                               self.data[col].mean() if self.data[col].dtype != 'object' else '-',
                               self.data[col].median() if self.data[col].dtype != 'object' else '-',
                               self.data[col].dtype,
                               self.data[col].isnull().sum(),
                               round(100 * self.data[col].isnull().sum()/self.data.shape[0], 2)
                               ]
        return pd.DataFrame.from_dict(self.stats, orient='index', columns=['Num Unique', 'First 10 unique', 'Min', 'Max', 'Mean', 'Median', 'Type', 'Missing total', 'Missing relative']).sort_values('Missing relative', ascending=False)

    def impute(self, columns, value):
        columns = columns if isinstance(self.data, list) else list(columns)
        for col in columns:
            self.data[col].fillna(value, inplace=True)    

    def iterative(self, columns, estimator):        
        ii = IterativeImputer(estimator=estimator(), max_iter=10, random_state=0)
        res = pd.DataFrame(ii.fit_transform(pd.DataFrame(self.data[columns])), columns=columns)
        for col in res.columns:
            self.data[col] = res[col]

    def plot(self):
        for col in self.data.columns:
            sns.histplot(self.data[col], color="dodgerblue", label="Compact")
    
    def process(self):
        for col in self.data.columns:
            if self.data[col].dtype=='object' or self.data[col].nunique() < 200:
                self.data[col].fillna('missing', inplace=True)
                l_enc = LabelEncoder()
                l_enc.fit(self.data[col].astype(str))
                self.data[col] = l_enc.transform(self.data[col].astype(str))        
                self.categorical_columns.append(col)
                self.categorical_dims[col] = len(l_enc.classes_)
            if self.data[col].nunique() >= 200:
                self.data.fillna(self.data[col].mean(), inplace=True)