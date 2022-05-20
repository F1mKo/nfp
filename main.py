from datetime import datetime
import pandas as pd
from model import run_model



config = {"input_file": "scenarios.xlsx",
          'sheet_name': 'augmentation',
          'scenario_number': '10733_1',
          'cycle_length': 7}  # settings


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df


now = datetime.now()
df = parse_data(config['input_file'], config['sheet_name'])
print(df)
result = run_model(df, config)
print('Total execution time', datetime.now() - now)


