# Import packages
import csv
from itertools import islice
from datetime import datetime

import pandas as pd
import numpy as np
import openpyxl
import copy

import gurobipy as gp
from gurobipy import GRB, quicksum
import grblogtools


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


def catch_case(database, scenario_id):
    return database[database.index.isin([scenario_id])]


def split_data(d):
    return [int(i) for i in d.split(';')]


def arc_param(arcs, param):
    result = {(i, j, t): param[i] if i + 1 == j else param[j] for (i, j, t) in arcs}
    return result


def preprocessing(case, cycle_len):
    # case data processing

    # distances
    distances = split_data(case[0])
    n = len(distances) + 1

    # crew sizes
    crew_size = split_data(case[1])

    # drivers set
    drivers = [i for i in range(0, n ** 2)]

    # Generate Nodes list
    nodes = [i for i in range(n + 1)]

    # forward/backward departure data
    if str(case[2]).isdigit():
        forward_departure = int(case[2])
    else:
        forward_departure = split_data(case[2])
    if str(case[3]).isdigit():
        backward_departure = int(case[3])
    else:
        backward_departure = split_data(case[3])
    departures = [forward_departure, backward_departure]

    # forward/backward Arc matrix
    arcs = arcs_creator(departures, distances, cycle_len)

    # unique time set
    t_set = set([item[2] for item in arcs])
    t_set = list(sorted(t_set))

    # crew size for each arc
    arc_crew_size = arc_param(arcs, crew_size)

    # arcs service durations
    t_a = arc_param(arcs, distances)

    # A_a_x and A_a_y set
    Aax = {(i, j, t): find_closest_arrive((i, j, t), arcs, t_a, 11, cycle_length) for (i, j, t) in arcs}
    Aay = {(i, j, t): find_closest_arrive((i, j, t), arcs, t_a, 24, cycle_length) for (i, j, t) in arcs}

    result = {
        'Nodes set': nodes,  # set of nodes in the network
        'Arcs set': arcs,  # set of arcs (works) to be served
        'Drivers set': drivers,  # set of drivers
        'Aax subset': Aax,  # set of arcs with the closest arrival time to departure arc a with daily rest
        'Aay subset': Aay,  # set of arcs with the closest arrival time to departure arc a with weekly rest
        'Akw set': arcs,  # set of arcs, which belongs to the week 𝑘
        'Akww set': arcs,  # set of arcs, which belongs to the double week 𝑘
        'time set': t_set,  # unique time set
        'arc crew size': arc_crew_size,  # crew size on the arc 𝑎 ∈ 𝐴
        'arcs service time': t_a  # arcs service durations

    }
    return result


def route_sim(departures, distances, cycle):
    result = []
    n = len(distances)
    time_limit = 24 * cycle
    for i in range(cycle):
        dep_forward = departures[0] + i * 24
        dep_backward = departures[1] + i * 24
        result.append([0, 1, dep_forward])
        result.append([n, n - 1, dep_backward])
        for j in range(n):
            dep_forward += distances[j]
            dep_backward += distances[n - j - 1]
            result.append([j, j + 1, dep_forward % time_limit])
            result.append([n - j, n - j - 1, dep_backward % time_limit])
    return result


def arcs_creator(departures, distances, cycle_len=7):
    # Generate forward/backward Arc matrix
    arcs = []
    if isinstance(departures[0], list) and isinstance(departures[1], list):
        for cur_dep in zip(departures[0], departures[1]):
            temp = route_sim(cur_dep, distances, cycle_len)
            arcs += temp
    else:
        arcs = route_sim(departures, distances, cycle_len)
    arcs = sorted(arcs, key=lambda item: item[2])
    return arcs


def find_closest_arrive(a_, arcs, arc_len, rest_time, days_total):  # 11 or 24 relax time duration
    result = []
    time_limit = days_total * 24
    if a_[2] >= rest_time:
        time = a_[2] - rest_time
    else:
        time = a_[2] - rest_time + time_limit
    t_closest = 2 * time_limit
    for a in arcs[::-1]:
        if a[1] == a_[0]:
            t_arrive = a[2] + arc_len[a[0], a[1], a[2]]
            if time >= t_arrive:
                t_between = time - t_arrive
            else:
                t_between = time + time_limit - t_arrive
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
#    print('rel_time', rest_time, 'ans', a_, '==', result)
    return result


def create_model(data):
    # Resource, nodes and arcs sets
    N = data['Nodes set']  # set of nodes in the network
    A = data['Arcs set']  # set of arcs (works) to be served
    D = data['Drivers set']  # set of drivers
    Aax = data['Aax subset']  # set of arcs with the closest arrival time to departure arc a with daily rest
    Aay = data['Aay subset']  # set of arcs with the closest arrival time to departure arc a with weekly rest
    Akw = data['Arcs set']  # set of arcs, which belongs to the week 𝑘
    Akww = data['Arcs set']  # set of arcs, which belongs to the double week 𝑘
    t_set = data['time set']  # unique time set
    print('t_set', t_set)
    # Constants
    c_a = data['arc crew size']  # crew size on the arc 𝑎 ∈ 𝐴
    t_a = data['arcs service time']  # arcs service durations

    # Declare and initialize model
    m = gp.Model('NFP')

    # Create decision variables for the NFP model
    da = [(d, i, j, t) for (i, j, t) in A for d in D]
    x_da = m.addVars(da, vtype=GRB.BINARY,
                     name='x_da')  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴, 0 otherwise
    y_da = m.addVars(da, vtype=GRB.BINARY,
                     name='y_da')  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴 and have a weekly rest on the end node of arc, 0 otherwise
    dit = [(d, i, t) for t in t_set for i in N for d in D]
    s_dit = m.addVars(dit, vtype=GRB.BINARY,
                      name='s_dit')  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is located in node 𝑖 ∈ 𝑁 at time 𝑡, 0 otherwise
    b_d = m.addVars(D, vtype=GRB.BINARY,
                    name='b_d')  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is selected, 0 otherwise

    # Create variables for convenient output
    work_d = m.addVars(D, vtype=GRB.CONTINUOUS, name='driver_work_duration')

    # Objective: maximize total matching score of all assignments
    m.setObjective(quicksum(b_d[i] for i in D), GRB.MINIMIZE)
    A_cycle = copy.deepcopy(A)
#    A_cycle.append(A[0])
    t_set_cycle = copy.deepcopy(t_set)
    t_set_cycle.append(t_set[0])

    # Create constraints
    driver_movement = m.addConstrs(((s_dit[d, i, t] + x_da[d, i, j, t] + y_da[d, i, j, t] ==
                                     quicksum(s_dit[d, i, t_set[k]] for k in range(len(t_set)) if t_set_cycle[k+1] == t)
                                     + quicksum((x_da[d, i1, j1, t1]) for (i1, j1, t1) in Aax[i, j, t]) + quicksum(
                (y_da[d, i2, j2, t2]) for (i2, j2, t2) in Aay[i, j, t])) for (i, j, t) in A_cycle
                                     for d in D), name='driver_movement')

    # Driver weekly work time definition and constraints
    driver_weekly_work_duration = m.addConstrs(
        (quicksum(t_a[i, j, t] * (x_da[d, i, j, t] + y_da[d, i, j, t]) for (i, j, t) in Akw) == work_d[d] for d in D),
        name='driver_wwd_definition')
    driver_wwd_constraints = m.addConstrs((work_d[d] <= 56 for d in D), name='driver_wwd_constraints')
    symmetry_breaking_wwd_constraints = m.addConstrs((work_d[D[i + 1]] <= work_d[D[i]] for i in range(len(D) - 1)),
                                                     name='symmetry_breaking_wwd_constraints')

    # Create crew size constraints
    crew_size_constraints = m.addConstrs(
        (quicksum(x_da[d, i, j, t] + y_da[d, i, j, t] for d in D) == c_a[i, j, t] for (i, j, t) in A),
        name='crew_size_constr')

    # Create weekly rest constraints
    weekly_rest_constraints = m.addConstrs((quicksum(y_da[d, i, j, t] for (i, j, t) in Akw) >= b_d[d] for d in D),
                                           name='weekly_rest_constraints')
    # Create weekly rest constraints
    dm_constraints = m.addConstrs((s_dit[d, i, t] + x_da[d, i, j, t] + y_da[d, i, j, t] == b_d[d]
                                   for (i, j, t) in A for d in D), name='dm_constraints')

    # Create weekly rest constraints
#    dm_constraints1 = m.addConstrs((quicksum(s_dit[d, i, t] for i in N) <= b_d[d]
#                                   for t in t_set for d in D), name='dm_constraints1')

    # Create driver selection definition
    driver_selection_definition = m.addConstrs(
        (quicksum(x_da[d, i, j, t] + y_da[d, i, j, t] for (i, j, t) in A) <= 10000 * b_d[d] for d in D),
        name='driver_selection_definition')

    # Create driver selection symmetry breaking constraints
    driver_selection_symmetry = m.addConstrs((b_d[D[i]] >= b_d[D[i + 1]] for i in range(len(D) - 1)),
                                             name='driver_selection_definition')

    # Save model for inspection
    m.write('NFP.lp')
    return (m)


now = datetime.now()
# Get scenario data
df = get_database('C:/Users/F1mKo/PycharmProjects/gurobi_conda/scenarios.xlsx')
cur_case = catch_case(df, '10733_1')
cycle_length = 7
data = preprocessing(cur_case.values[0], cycle_length)

# Declare and initialize model
m = create_model(data)

# Run optimization engine
m.optimize()

# Display optimal values of decision variables
# print(m.getVars())
for v in m.getVars():
    if v.x > 1e-6:
        print(v.varName, v.x)

'''
'''
print('Total execution time', datetime.now() - now)
# Display optimal total matching score
# print('Total matching score: ', m.objVal)

varInfo = [(v.varName, v.X) for v in m.getVars() if v.X > 0]

# Write to csv
with open('model_out.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(varInfo)


