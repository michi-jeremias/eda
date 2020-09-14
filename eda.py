import pandas as pd


class DataMangler:
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def drop(self, col):
        if not isinstance(list, col):
            col = [col]

        try:
            for c in col:
                self.data.drop(c, inplace=True)
        except IndexError:
            print('Invalid index')

    def eda(self):
        for col in self.data.columns:
            print(f"Col '{col}': {self.data[col].nunique()}")
