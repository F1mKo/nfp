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

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


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


def plot_network(arcs, distances):
    fig = plt.figure()
    #    ax = plt.axes(projection='3d')
    ax = plt.axes()
    for a in arcs:
        ax.plot([a[0], a[1]], [a[2], a[2] + distances[a[0], a[1], a[2]] % 168], 'gray')
    # Data for three-dimensional scattered points
    #    zdata = [item[2] for item in arcs]
    #    xdata = [item[0] for item in arcs]
    #    ydata = [item[1] for item in arcs]
    #    ax.plot3D(xdata, ydata, zdata, 'gray')
    #    ax.plot(ydata, zdata, 'gray')
    #    ax.view_init(0, 90)
    plt.show()


def preprocessing(case, cycle_len):
    # case data processing

    # distances
    distances = split_data(case[0])
    n = len(distances) + 1

    # crew sizes
    crew_size = split_data(case[1])

    # drivers set
    drivers = [i for i in range(0, n ** 2 // 2)]

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

    # crew size for each arc
    arc_crew_size = arc_param(arcs, crew_size)

    # arcs service durations
    t_a = arc_param(arcs, distances)

    # plot_network(arcs, t_a)

    # unique time set
    t_set = set([item[2] for item in arcs])
    t_set = list(sorted(t_set))

    # A_a_x and A_a_y set
    time_limit = cycle_len * 24
    Aax = {(i, j, t): find_closest_arrive((i, j, t), arcs, t_a, 11, time_limit) for (i, j, t) in arcs}
    Aay = {(i, j, t): find_closest_arrive((i, j, t), arcs, t_a, 24, time_limit) for (i, j, t) in arcs}

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


def route_sim(departures, distances, cycle_len):
    result_forward = []
    result_backward = []
    n = len(distances)
    time_limit = 24 * cycle_len
    for i in range(cycle_len):
        dep_forward = departures[0] + i * 24
        dep_backward = departures[1] + i * 24
        result_forward.append([0, 1, dep_forward])
        result_backward.append([n, n - 1, dep_backward])
        for j in range(1, n):
            dep_forward += distances[j - 1]
            dep_backward += distances[n - j]
            result_forward.append([j, j + 1, dep_forward % time_limit])
            result_backward.append([n - j, n - j - 1, dep_backward % time_limit])
    return result_forward + result_backward


def arcs_creator(departures, distances, cycle_len=7):
    # Generate forward/backward Arc matrix
    arcs = []
    if isinstance(departures[0], list) and isinstance(departures[1], list):
        for cur_deps in zip(departures[0], departures[1]):
            temp = route_sim(cur_deps, distances, cycle_len)
            arcs += temp
    else:
        arcs = route_sim(departures, distances, cycle_len)
    #    arcs = sorted(arcs, key=lambda item: item[2])
    # print(arcs)
    return arcs


def find_closest_arrive(a_, arcs, arc_len, rest_time, time_limit):  # 11 or 24 relax time duration
    result = []
    time = a_[2]
    t_closest = 2 * time_limit
    for a in arcs[::-1]:
        if a[1] == a_[0]:
            t_arrive = (a[2] + arc_len[a[0], a[1], a[2]] + rest_time) % time_limit
            if t_arrive <= time:
                t_between = time - t_arrive
            else:
                t_between = time - t_arrive + time_limit
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
    # print('rel_time', rest_time, 'ans', a_, '==', result)
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
    x_da = {(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                   name="x_{0}_{1}_{2}_{3}".format(d, i, j, t))
            for (i, j, t) in A for d in
            D}  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴, 0 otherwise
    y_da = {(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                   name="y_{0}_{1}_{2}_{3}".format(d, i, j, t))
            for (i, j, t) in A for d in
            D}  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴 and have a weekly rest on the end node of arc, 0 otherwise

    s_dit = {(d, i, t): m.addVar(vtype=GRB.BINARY,
                                 name="s_{0}_{1}_{2}".format(d, i, t))
             for t in t_set for i in N for d in
             D}  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is located in node 𝑖 ∈ 𝑁 at time 𝑡, 0 otherwise
    b_d = {d: m.addVar(vtype=GRB.BINARY, name="b_{0}".format(d)) for d in
           D}  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is selected, 0 otherwise

    # Create variables for convenient output
    work_d = {d: m.addVar(vtype=GRB.CONTINUOUS, name="driver_{0}_work_duration".format(d)) for d in D}

    # Objective: maximize total matching score of all assignments
    m.setObjective(quicksum(b_d[i] for i in D), GRB.MINIMIZE)

    # Create constraints
    driver_movement = {(d, i, j, t):
                           m.addConstr(s_dit[d, i, t] + x_da[d, i, j, t] + y_da[d, i, j, t] ==
                                       quicksum(
                                           s_dit[d, i, t_set[k - 1]] for k in range(len(t_set)) if t_set[k] == t)
                                       + quicksum((x_da[d, i1, j1, t1]) for (i1, j1, t1) in Aax[i, j, t]) +
                                       quicksum((y_da[d, i2, j2, t2]) for (i2, j2, t2) in Aay[i, j, t]),
                                       name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                       for (i, j, t) in A for d in D}

#    driver_movement1 = {(d, i):
#                            m.addConstr(quicksum(
#                                (x_da[d, i1, j1, t1] + y_da[d, i1, j1, t1]) for (i1, j1, t1) in A if i1 == i) ==
#                                        quicksum((x_da[d, i2, j2, t2] + y_da[d, i2, j2, t2]) for (i2, j2, t2) in A if
#                                                 j2 == i),
#                                        name="driver_movement1_{0}_{1}".format(d, i))
#                        for d in D for i in N}

    # Driver weekly work time definition and constraints
    driver_weekly_work_duration = {d: m.addConstr(
        quicksum(t_a[i, j, t] * (x_da[d, i, j, t] + y_da[d, i, j, t]) for (i, j, t) in Akw) == work_d[d],
        name="driver_wwd_definition_{0}".format(d))
        for d in D}

    driver_wwd_constraints = {d: m.addConstr(work_d[d] <= 56,
                                             name="driver_wwd_constraints_{0}".format(d))
                              for d in D}

    symmetry_breaking_wwd_constraints = {D[i]: m.addConstr(work_d[D[i + 1]] <= work_d[D[i]],
                                                           name="symmetry_breaking_wwd_constraints_{0}".format(D[i]))
                                         for i in range(len(D) - 1)}

    # Create crew size constraints
    crew_size_constraints = {
        (i, j, t): m.addConstr(quicksum(x_da[d, i, j, t] + y_da[d, i, j, t] for d in D) == c_a[i, j, t],
                               name="crew_size_constr_{0}_{1}_{2}".format(i, j, t))
        for (i, j, t) in A}

    # Create weekly rest constraints
    weekly_rest_constraints = {d: m.addConstr(quicksum(y_da[d, i, j, t] for (i, j, t) in Akw) >= b_d[d],
                                              name="weekly_rest_constraints_{0}".format(d))
                               for d in D}

    #   Create driver selection definition
    driver_selection_definition = {
        d: m.addConstr(quicksum(x_da[d, i, j, t] + y_da[d, i, j, t] for (i, j, t) in A) <= 10000 * b_d[d],
                       name="driver_selection_definition_{0}".format(d))
        for d in D}

    # Create driver selection symmetry breaking constraints
    symmetry_breaking_ds_constraints = {D[i]: m.addConstr(b_d[D[i]] >= b_d[D[i + 1]],
                                                          name="symmetry_breaking_ds_constraints_{0}".format(D[i]))
                                        for i in range(len(D) - 1)}

    # Additional constraints
#    dm_constraints = m.addConstrs((s_dit[d, i, t] + quicksum(x_da[d, i1, j1, t1] + y_da[d, i1, j1, t1] for (i1, j1, t1) in A if (t1 == t and i1 == i)) == b_d[d]
#                                    for t in t_set for i in N for d in D), name='dm_constraints')
    #    dm_constraints1 = m.addConstrs((quicksum(s_dit[d, i, t] for i in N) <= b_d[d]
    #                                    for t in t_set for d in D), name='dm_constraints1')
    #    dm_constraints2 = m.addConstrs((quicksum(x_da[d, i1, j1, t1] for (i1, j1, t1) in Aax[i, j, t]) +
    #                                    quicksum(y_da[d, i2, j2, t2] for (i2, j2, t2) in Aay[i, j, t]) <= b_d[d]
    #                                    for (i, j, t) in A for d in D), name='dm_constraints2')

    # Save model for inspection
    m.write('NFP.lp')
    return m


now = datetime.now()
# Get scenario data
df = get_database('C:/Users/F1mKo/PycharmProjects/gurobi_conda/scenarios.xlsx')
cur_case = catch_case(df, '10733_1')
cycle_length = 7
case_data = preprocessing(cur_case.values[0], cycle_length)

# Declare and initialize model
model = create_model(case_data)

# Run optimization engine
model.optimize()
# model.computeIIS()
# model.write("model.ilp")
# Display optimal values of decision variables
# print(m.getVars())
for v in model.getVars():
    if v.x > 1e-6:
        print(v.varName, v.x)

'''
'''
print('Total execution time', datetime.now() - now)
# Display optimal total matching score
# print('Total matching score: ', m.objVal)

varInfo = [(v.varName.split('_')[1], v.varName.split('_')[-1], v.varName, v.X) for v in model.getVars() if v.X > 0]

# Write to csv
with open('model_out.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(varInfo)
