# Import packages
from itertools import islice
from datetime import datetime

import pandas as pd
import numpy as np
import openpyxl
from model import run_model


def get_database(path):
    wb = openpyxl.load_workbook(filename=path)
    sheet = wb['augmentation']
    # Put the sheet values in `data`
    data = sheet.values
    # Indicate the columns in the sheet values
    cols = next(data)[1:]
    # Convert your data to a list
    data = list(data)[:96]
    # Read in the data at index 0 for the indices
    idx = [r[0] for r in data]
    # Slice the data at index 1
    data = (islice(r, 1, None) for r in data)
    # Make your DataFrame
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.drop(df.columns[[0, 5, 6]], axis=1, inplace=True)
    df.index.name = 'ID'
    return df


def split_data(data):
    return [int(i) for i in data.split(';')]


def process_case(database, scenario_id, cycle_len):
    case = list(database[database.index.isin([scenario_id])].values[0])
    for i in range(len(case)):
        if str(case[i]).isdigit():
            case[i] = int(case[i])
        else:
            case[i] = split_data(case[i])
    case.append(cycle_len)
    print(case)
    return case



now = datetime.now()
# Get scenario data
df = get_database('scenarios.xlsx')
scenario_number = '10733_1'
cycle_length = 7
#case = process_case(df, scenario_number, cycle_length)
case = [[7, 12, 8], [1, 1, 1], 7, 7, 7]
result = run_model(case)

#nfp = NFPmodel(cur_case, cycle_length)

# Run optimization engine
#nfp.model.optimize()
#model.computeIIS()
#model.write("model.ilp")

#print('Total execution time', datetime.now() - now)
#result_csv(nfp.model)

# Display optimal values of decision variables
# print(m.getVars())
#for v in model.getVars():
#    if v.x > 1e-6:
#        print(v.varName, v.x)
# Display optimal total matching score
# print('Total matching score: ', m.objVal)

