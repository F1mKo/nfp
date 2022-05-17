# Import packages
import gurobipy as gp
from gurobipy import Model, tuplelist, tupledict, GRB, quicksum
import grblogtools
import matplotlib.pyplot as plt
import csv


class ModelData:
    def __init__(self, case):
        # distances between nodes i and i+1
        self.distances = tuplelist(case[0])
        self.n = len(self.distances)

        # get crew size values
        self.crew_size = tuplelist(case[1])

        # forward/backward departure data
        self.departures = [case[2], case[3]]

        # generate nodes set N
        self.nodes = tuplelist(i for i in range(self.n + 1))  # set of nodes in the network
        # generate drivers set D
        self.drivers = tuplelist(d for d in range(0, 3 * self.n))  # set of drivers
        self.cycle_length = case[4]
        self.time_limit = self.cycle_length * 24

        # forward/backward Arc matrix with departure and arriving info
        self.arcs_dep, self.arcs_arr = self.arcs_creator()  # set of arcs (works) to be served

        # crew size for each arc
        self.c_a = arc_param(self.arcs_dep, self.crew_size)

        # arcs service durations
        self.t_a = arc_param(self.arcs_dep, self.distances)

        self.plot_network()

        # unique time set T
        uniq_time_set = set([item[2] for item in self.arcs_dep])
        self.t_set = tuplelist(sorted(uniq_time_set))

        # A_a_x and A_a_y set
        self.Aax = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 11, self.time_limit)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        self.Aay = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 24, self.time_limit)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest
        self.Aax_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 11), self.time_limit)
            for (i, j, t) in self.arcs_dep})
        self.Aay_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 24), self.time_limit)
            for (i, j, t) in self.arcs_dep})
        self.Akw = self.arcs_dep  # set of arcs, which belongs to the week 𝑘
        self.Akww = self.arcs_dep  # set of arcs, which belongs to the double week 𝑘

    def plot_network(self):
        ax = plt.axes()
        for a in self.arcs_dep:
            ax.plot([a[0], a[1]], [a[2], (a[2] + self.distances[min(a[0], a[1])]) % self.time_limit], 'blue')
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Time (hours)')
        plt.show()

    def arcs_creator(self):
        # Generate forward/backward Arc matrix
        arcs_dep = []
        arcs_arr = []
        if isinstance(self.departures[0], list) and isinstance(self.departures[1], list):
            for cur_deps in zip(self.departures[0], self.departures[1]):
                temp = route_sim(cur_deps, self.distances, self.cycle_length)
                arcs_dep += temp[0]
                arcs_arr += temp[1]
        else:
            arcs_dep, arcs_arr = route_sim(self.departures, self.distances, self.cycle_length)
        #    arcs_dep = sorted(arcs_dep, key=lambda item: item[2])
        # print(arcs_dep)
        return tuplelist(arcs_dep), tuplelist(arcs_arr)


class ModelVars:
    def __init__(self):
        self.x_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴, 0 otherwise
        self.y_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴 and have a weekly rest on the end node of arc, 0 otherwise
        self.s_dit = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is located in node 𝑖 ∈ 𝑁 at time 𝑡, 0 otherwise
        self.b_d = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is selected, 0 otherwise
        self.work_d = tupledict()  # driver total work duration

    def write_values(self):
        for attr in tuple(vars(self)):
            if not attr.startswith("__"):
                setattr(self, attr, {key: val.X for (key, val) in getattr(self, attr).items()})


def add_variables(m: Model, data: ModelData, v: ModelVars):
    v.x_da = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                               name="x_{0}_{1}_{2}_{3}".format(d, i, j, t))
                        for d in data.drivers for (i, j, t) in data.arcs_dep})
    v.y_da = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                               name="y_{0}_{1}_{2}_{3}".format(d, i, j, t))
                        for d in data.drivers for (i, j, t) in data.arcs_dep})
    v.s_dit = tupledict({(d, i, t): m.addVar(vtype=GRB.BINARY,
                                             name="s_{0}_{1}_{2}".format(d, i, t))
                         for d in data.drivers for i in data.nodes for t in data.t_set})
    v.b_d = tupledict({d: m.addVar(vtype=GRB.BINARY, name="b_{0}".format(d)) for d in data.drivers})
    v.work_d = tupledict({d: m.addVar(vtype=GRB.CONTINUOUS, name="driver_{0}_work_duration".format(d)) for d in
                          data.drivers})


def add_constraints(m: Model, data: ModelData, v: ModelVars):
    driver_movement = tupledict({(d, i, t): m.addConstr(v.s_dit[d, i, t] + quicksum(
                                         (quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax_inv[ik, jk, tk]) +
                                          quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay_inv[ik, jk, tk]))
                                         for (ik, jk, tk) in data.arcs_dep if ik == i and tk == t)
                                                 == quicksum(
                                         v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set)) if
                                         data.t_set[k] == t)
                                                 + quicksum(
                                         (quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[ik, jk, tk]) +
                                          quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[ik, jk, tk]))
                                         for (ik, jk, tk) in data.arcs_dep if jk == i and tk == t),
                                                 name="driver_movement_{0}_{1}_{2}".format(d, i, t))
                                 for d in data.drivers for i in data.nodes for t in data.t_set})

#    driver_movement1 = tupledict({(d, i, j, t):
#                            m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] == quicksum(
#                                v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set)) if data.t_set[k] == t)
#                                        + quicksum((v.x_da[d, i1, j1, t1]) for (i1, j1, t1) in data.Aax[i, j, t]) +
#                                        quicksum((v.y_da[d, i2, j2, t2]) for (i2, j2, t2) in data.Aay[i, j, t]),
#                                        name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
#                        for d in data.drivers for (i, j, t) in data.arcs_dep})

    # Driver weekly work time definition and constraints
    driver_weekly_work_duration = tupledict({d: m.addConstr(
        quicksum(data.t_a[i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (i, j, t) in data.Akw) == v.work_d[d],
        name="driver_wwd_definition_{0}".format(d))
        for d in data.drivers})

    driver_wwd_constraints = tupledict({d: m.addConstr(v.work_d[d] <= 56,
                                                       name="driver_wwd_constraints_{0}".format(d)) for d in
                                        data.drivers})

    # Create crew size constraints
    crew_size_constraints = tupledict({
        (i, j, t): m.addConstr(
            quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for d in data.drivers) == data.c_a[i, j, t],
            name="crew_size_constr_{0}_{1}_{2}".format(i, j, t))
        for (i, j, t) in data.arcs_dep})

    # Create weekly rest constraints
    weekly_rest_constraints = tupledict(
        {d: m.addConstr(quicksum(v.y_da[d, i, j, t] for (i, j, t) in data.Akw) >= v.b_d[d],
                        name="weekly_rest_constraints_{0}".format(d)) for d in
         data.drivers})

    #   Create driver selection definition
    driver_selection_definition = tupledict({d: m.addConstr(
        quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for (i, j, t) in data.arcs_dep) <= 10000 * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})


def add_symmetry_breaking_constr(m: Model, data: ModelData, v: ModelVars):
    # Create driver work_time symmetry breaking constraints
    symmetry_breaking_wwd_constraints = {
        data.drivers[i]: m.addConstr(v.work_d[data.drivers[i + 1]] <= v.work_d[data.drivers[i]],
                                     name="symmetry_breaking_wwd_constraints_{0}".format(
                                         data.drivers[i]))
        for i in range(len(data.drivers) - 1)}
    # Create driver selection symmetry breaking constraints
    symmetry_breaking_ds_constraints = {
        data.drivers[i]: m.addConstr(v.b_d[data.drivers[i]] >= v.b_d[data.drivers[i + 1]],
                                     name="symmetry_breaking_ds_constraints_{0}".format(
                                         data.drivers[i]))
        for i in range(len(data.drivers) - 1)}


def add_objective(m: Model, data: ModelData, v: ModelVars):
    m.setObjective(quicksum(v.b_d[i] for i in data.drivers), GRB.MINIMIZE)


def result_csv(m: Model):
    columns = ['Driver', 'i', 'time', 'variable', 'value']
    varInfo = [(v.varName.split('_')[1], v.varName.split('_')[2], v.varName.split('_')[-1], v.varName, v.X) for v in
               m.getVars() if (v.X > 0 and len(v.varName.split('_')) > 2)]

    # Write to csv
    with open('model_out.csv', 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(columns)
        wr.writerows(varInfo)


def run_model(case):
    # Declare and initialize model
    m = Model('NFP')
    data = ModelData(case)
    v = ModelVars()

    add_variables(m, data, v)
    add_constraints(m, data, v)
    add_symmetry_breaking_constr(m, data, v)
    add_objective(m, data, v)

    m.setParam('Heuristics', 0.5)
    m.setParam('MIPFocus', 1)
    m.setParam('Threads', 8)

    # m.setParam('SolutionLimit', 1)
    m.update()
    m.write('nfp.lp')
    #m.read('nfp.sol')
    # m.computeIIS()
    # m.write('inf.ilp')
    m.optimize()
    m.write('nfp.sol')
#    v.write_values()
    result_csv(m)
    return m


def arc_param(arcs, param):
    return tupledict({(i, j, t): param[i] if i + 1 == j else param[j] for (i, j, t) in arcs})


def route_sim(departures, distances, cycle_len):
    dep_forward = []
    dep_backward = []
    n = len(distances)
    time_limit = 24 * cycle_len
    arr_forward = []
    arr_backward = []
    for i in range(cycle_len):
        dep_forward_time = departures[0] + i * 24
        dep_backward_time = departures[1] + i * 24
        dep_forward.append([0, 1, dep_forward_time])
        dep_backward.append([n, n - 1, dep_backward_time])
        arr_forward.append([0, 1, dep_forward_time + distances[0]])
        arr_backward.append([n, n - 1, dep_backward_time + distances[-1]])
        for j in range(1, n):
            dep_forward_time += distances[j - 1]
            dep_backward_time += distances[n - j]
            dep_forward.append([j, j + 1, dep_forward_time % time_limit])
            dep_backward.append([n - j, n - j - 1, dep_backward_time % time_limit])
            arr_forward.append([j, j + 1, (dep_forward_time + distances[j]) % time_limit])
            arr_backward.append([n - j, n - j - 1, (dep_backward_time + distances[n - j - 1]) % time_limit])
    return dep_forward + dep_backward, arr_forward + arr_backward


def find_closest_arrive(a_, arcs_arr, arc_len, rest_time, time_limit):  # 11 or 24 relax time duration
    result = []
    time = a_[2] - rest_time
    t_closest = 2 * time_limit
    for a in arcs_arr[::-1]:
        if a[1] == a_[0]:
            if a[2] <= time:
                t_between = time - a[2]
            else:
                t_between = time - a[2] + time_limit
            if t_between <= t_closest:
                arc_dep_time = (a[2] - arc_len[min(a[0], a[1])]) if a[2] >= arc_len[min(a[0], a[1])] else \
                    (a[2] - arc_len[min(a[0], a[1])] + time_limit)
                if t_between < t_closest:
                    t_closest = t_between
                    result = [[a[0], a[1], arc_dep_time]]
                else:
                    result.append([a[0], a[1], arc_dep_time])

    # print('rel_time', rest_time, 'ans', a_, '==', result)
    return result


def find_closest_depart(a_, arcs_dep, rest_time, time_limit):  # 11 or 24 relax time duration
    result = []
    time = a_[2] + rest_time
    t_closest = 2 * time_limit
    for a in arcs_dep:
        if a[0] == a_[1]:
            if a[2] >= time:
                t_between = a[2] - time
            else:
                t_between = a[2] - time + time_limit
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
    # print('rel_time', rest_time, 'ans', a_, '==', result)
    return result
