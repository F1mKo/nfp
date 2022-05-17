from datetime import datetime
import pandas as pd
from model import run_model


def split_data(data):
    return [int(i) for i in data.split(';')]


def process_case(database, scenario_id, cycle_len):
    case = database[database.index.isin([scenario_id])].values.tolist()[0]
    print(case)
    for i in range(len(case)):
        if str(case[i]).isdigit():
            case[i] = int(case[i])
        else:
            case[i] = split_data(case[i])
    case.append(cycle_len)
    print(case)
    return case


config = {"input_file": "scenarios.xlsx",
          'sheet_name': 'augmentation',
          'scenario_number': '10733_1',
          'cycle_length': 7}  # вынес файлы наружу.


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df

now = datetime.now()
# Get scenario data
df = parse_data(config['input_file'], config['sheet_name'])
#print(df)
case = process_case(df, config['scenario_number'], config['cycle_length'])
#case = [[7, 12, 8], [1, 1, 1], 7, 7, 7]
result = run_model(case)
print('Total execution time', datetime.now() - now)


