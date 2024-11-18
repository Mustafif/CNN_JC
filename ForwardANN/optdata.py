import pandas as pd

class OptionData:
    def __init__(self, day_num, opt_file=None):
        if opt_file is None:
            self.filename = f"Day{day_num}.csv"
        else:
            self.filename = opt_file


    def call(self):
        df = pd.read_csv(self.filename)
        call_prices = df[df['Type'] == 'Call'].iloc[:, 2:].to_numpy()
        return call_prices
    def put(self):
        df = pd.read_csv(self.filename)
        put_prices = df[df['Type'] == 'Put'].iloc[:, 2:].to_numpy()
        return put_prices
