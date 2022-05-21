from gurobipy import Model, tuplelist, tupledict, GRB, quicksum
import matplotlib.pyplot as plt
import csv
import random


class ModelData:
    def __init__(self, case_db, config):
        """
        ModelData --- class for data processing. It's used in model definition.
        :param case_db: scenarios database
        :param config:  run configurations
        """
        # catch the case run parameters
        self.case_id = config['scenario_number']
        self.cycle_length = config['cycle_length']

        # calculation of time horizon length corresponding to cycle length
        self.n_weeks = self.cycle_length // 7
        if self.n_weeks > 1:
            self.week_num = tuplelist(range(self.n_weeks))
            self.time_limit = tuplelist(
                [((i + 1) / self.n_weeks) * 24 * self.cycle_length for i in self.week_num])
        else:
            self.week_num = 0
            self.time_limit = 24 * self.cycle_length

        # catch distances between nodes i and i+1
        self.distances = self.cell_reader(case_db, 'Участки')

        # calculation of total road fragments amount
        self.n = len(self.distances)

        # catch crew size values
        self.crew_size = self.cell_reader(case_db, 'Водители')

        # generate nodes set N
        self.nodes = tuplelist(i for i in range(self.n + 1))  # set of nodes in the network

        # generate drivers set D
        self.drivers = tuplelist(d for d in range(0, 4 * self.n))  # set of drivers

        # catch forward/backward departure data
        self.departures = [self.cell_reader(case_db, 'Выезды прямо'),
                           self.cell_reader(case_db, 'Выезды обратно')]

        # generate forward/backward Arc matrix with departure and arriving info
        self.arcs_dep, self.arcs_arr = self.arcs_network_creator()  # set of arcs (works) to be served

        # crew size for each arc
        self.c_a = arc_param(self.arcs_dep, self.crew_size)

        # arcs service durations
        self.t_a = arc_param(self.arcs_dep, self.distances)

        # unique time set T
        uniq_time_set = set([item[2] for item in self.arcs_dep])
        self.t_set = tuplelist(sorted(uniq_time_set))

        # A_a_x and A_a_y set
        self.Aax = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 11, self.get_last_elem(self.time_limit))
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        self.Aay = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 24, self.get_last_elem(self.time_limit))
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest
        self.Aax_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 11), self.get_last_elem(self.time_limit))
            for (i, j, t) in self.arcs_dep})
        self.Aay_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 24), self.get_last_elem(self.time_limit))
            for (i, j, t) in self.arcs_dep})
        self.Akw = self.arcs_week_subset(week='single')
        self.Akww = self.arcs_week_subset()  # set of arcs, which belongs to the double week 𝑘

    def arcs_week_subset(self, week='single'):
        """
        get arc service time according to the week
        :param week: rule of arcs subset definition
        :return: set of arcs, which belongs to the week 𝑘 (𝑘 =[0, 1] for 'single' week, k=0 for 'double')
        """
        result = {}
        if week == 'single' and self.n_weeks > 1:
            for k in self.week_num:
                for (i, j, t) in self.arcs_dep:
                    if t < self.time_limit[k] and (k == 0 or self.time_limit[k - 1] < t):
                        result[k, i, j, t] = (self.time_limit[k] - t
                                              if t + self.t_a[i, j, t] > self.time_limit[k] else self.t_a[i, j, t])
                        if t + self.t_a[i, j, t] > self.time_limit[k]:
                            result[self.week_num[k - 1], i, j, t] = t + self.t_a[i, j, t] - self.time_limit[k]
        else:
            result = {(0, i, j, t): self.t_a[i, j, t] for (i, j, t) in self.arcs_dep}
        return tupledict(result)

    def arcs_network_creator(self):
        """
        Generate forward/backward Arc matrix
        :return:
            tuplelist(arcs_dep) --- main arcs set with departure times
            tuplelist(arcs_arr) --- additional arcs set with arrival times to simplify calculations
            in closest arrive function
        """
        arcs_dep = []
        arcs_arr = []
        if isinstance(self.departures[0], list) and isinstance(self.departures[1], list):
            for cur_deps in zip(self.departures[0], self.departures[1]):
                temp = route_sim(cur_deps, self.distances, self.cycle_length)
                arcs_dep += temp[0]
                arcs_arr += temp[1]
        else:
            arcs_dep, arcs_arr = route_sim(self.departures, self.distances, self.cycle_length)
        # arcs_dep = sorted(arcs_dep, key=lambda item: item[2])
        # print(arcs_dep)
        return tuplelist(arcs_dep), tuplelist(arcs_arr)

    def cell_reader(self, case_db, cell_name):
        """
        Catches the cell_name values in case_db
        :param case_db: scenarios database
        :param cell_name: cell column name
        :return:
            tuplelist(result) if result is array
            result if result is number
        """
        result, var_type = self.split_data(case_db.loc[[self.case_id], cell_name].values[0])
        return tuplelist(result) if var_type == 'array' else result

    @staticmethod
    def get_last_elem(parameter):
        """
        returns last element of list or number, if input is integer
        :param parameter:
        :return:
        """
        return parameter if isinstance(parameter, int) else parameter[-1]

    @staticmethod
    def split_data(data):
        """
        Checks data structure of cell
        :param data: data from the database cell
        :return:
            number if cell contains only one number
            array if cell contains more than one number
        """
        if str(data).isdigit():
            return int(data), 'number'
        else:
            return [int(i) for i in data.split(';')], 'array'


class ModelVars:
    def __init__(self):
        """
        ModelVars --- class for variable definition. It stores all variables for convenient use in model.
        """
        self.x_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴, 0 otherwise
        self.y_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴 and have a weekly rest on the end node of arc, 0 otherwise
        self.s_dit = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is located in node 𝑖 ∈ 𝑁 at time 𝑡, 0 otherwise
        self.b_d = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is selected, 0 otherwise
        self.w_work_d = tupledict()  # driver weekly work duration
        self.ww_work_d = tupledict()  # driver double week work duration


def add_variables(m: Model, data: ModelData, v: ModelVars):
    """
    Defines variables in model according to data.
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    v.b_d = tupledict({d: m.addVar(vtype=GRB.BINARY, name="b_{0}".format(d)) for d in data.drivers})

    v.x_da = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                               name="x_{0}_{1}_{2}_{3}".format(d, i, j, t))
                        for d in data.drivers for (i, j, t) in data.arcs_dep})

    v.y_da = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                               name="y_{0}_{1}_{2}_{3}".format(d, i, j, t))
                        for d in data.drivers for (i, j, t) in data.arcs_dep})

    v.s_dit = tupledict({(d, i, t): m.addVar(vtype=GRB.BINARY,
                                             name="s_{0}_{1}_{2}".format(d, i, t))
                         for d in data.drivers for i in data.nodes for t in data.t_set})

    # driver single/double week work duration
    if data.n_weeks > 1:
        v.w_work_d = tupledict(
            {(k, d): m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}_{1}".format(k, d)) for d in
             data.drivers for k in data.week_num})

        v.ww_work_d = tupledict(
            {d: m.addVar(vtype=GRB.CONTINUOUS, name="d2wwd_{0}".format(d)) for d in
             data.drivers})
    else:
        v.w_work_d = tupledict(
            {(0, d): m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}".format(d)) for d in
             data.drivers})


def add_constraints(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    # Driver movement definition
    driver_movement = tupledict({(d, i, j, t):
                                     m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] == quicksum(
                                         v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set)) if
                                         data.t_set[k] == t)
                                                 + quicksum(
                                         (v.x_da[d, i1, j1, t1]) for (i1, j1, t1) in data.Aax[i, j, t]) +
                                                 quicksum(
                                                     (v.y_da[d, i2, j2, t2]) for (i2, j2, t2) in data.Aay[i, j, t]),
                                                 name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                                 for d in data.drivers for (i, j, t) in data.arcs_dep})

    # Driver weekly work time definition and constraints
    if data.n_weeks > 1:
        driver_weekly_work_duration = tupledict({d: m.addConstr(
            quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if
                     k == ki) == v.w_work_d[ki,
                                            d],
            name="driver_w_wd_definition_{0}_{1}".format(ki, d))
            for d in data.drivers for ki in data.week_num})

        driver_2weekly_work_duration = tupledict({d: m.addConstr(
            quicksum(data.Akww[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akww) ==
            v.ww_work_d[
                d],
            name="driver_2w_wd_definition_{0}".format(d))
            for d in data.drivers for ki in data.week_num})

        driver_w_wd_constraints = tupledict({(k, d): m.addConstr(v.w_work_d[k, d] <= 56,
                                                                 name="driver_w_wd_constraints_{0}_{1}".format(k, d))
                                             for (k, d) in v.w_work_d})

        driver_ww_wd_constraints = tupledict({d: m.addConstr(v.ww_work_d[d] <= 90,
                                                             name="driver_2w_wd_constraints_{0}".format(d))
                                              for d in data.drivers})

    else:
        driver_weekly_work_duration = tupledict({d: m.addConstr(
            quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw) ==
            v.w_work_d[0, d], name="driver_w_wd_definition_0_{0}".format(d))
            for d in data.drivers})

        driver_w_wd_constraints = tupledict({d: m.addConstr(v.w_work_d[0, d] <= 56,
                                                            name="driver_w_wd_constraints_{0}".format(d))
                                             for d in data.drivers})

    # Create crew size constraints
    crew_size_constraints = tupledict({
        (i, j, t): m.addConstr(
            quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for d in data.drivers) == data.c_a[i, j, t],
            name="crew_size_constr_{0}_{1}_{2}".format(i, j, t))
        for (i, j, t) in data.arcs_dep})

    # Create weekly rest constraints
    if data.n_weeks > 1:
        weekly_rest_constraints = tupledict(
            {d: m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw if k == ki) >= v.b_d[d],
                            name="weekly_rest_constraints_{0}".format(d)) for d in
             data.drivers for ki in data.week_num})
    else:
        weekly_rest_constraints = tupledict(
            {d: m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw) >= v.b_d[d],
                            name="weekly_rest_constraints_{0}".format(d)) for d in
             data.drivers})

    #   Create driver selection definition
    driver_selection_definition = tupledict({d: m.addConstr(
        quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for (i, j, t) in data.arcs_dep) <= 10000 * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})

    #   Create driver selection definition
    driver_selection_definition1 = tupledict({d: m.addConstr(
        quicksum(v.s_dit[d, i, t] for i in data.nodes for t in data.t_set) <= 10000 * v.b_d[d],
        name="driver_selection_definition1_{0}".format(d))
        for d in data.drivers})


def add_symmetry_breaking_constr(m: Model, data: ModelData, v: ModelVars):
    """
    Defines an additional symmetry breaking constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    # Create driver work_time symmetry breaking constraints
    # symmetry_breaking_wwd_constraints = {
    #     data.drivers[i]: m.addConstr(v.work_d[data.drivers[i + 1]] <= v.work_d[data.drivers[i]],
    #                                  name="symmetry_breaking_wwd_constraints_{0}".format(
    #                                      data.drivers[i]))
    #     for i in range(len(data.drivers) - 1)}
    # Create driver selection symmetry breaking constraints
    symmetry_breaking_ds_constraints = {
        data.drivers[i]: m.addConstr(v.b_d[data.drivers[i]] >= v.b_d[data.drivers[i + 1]],
                                     name="symmetry_breaking_ds_constraints_{0}".format(
                                         data.drivers[i]))
        for i in range(len(data.drivers) - 1)}


def add_objective(m: Model, data: ModelData, v: ModelVars):
    """
    Defines an objective function for model
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    m.setObjective(quicksum(v.b_d[i] for i in data.drivers), GRB.MINIMIZE)


def result_csv(m: Model,):
    """
    Catches variables values from model optimization results. Creates a csv-type file with determined columns
    :param m: Model class instance
    :return: None
    """
    columns = ['Driver', 'i', 'j', 'time', 'variable', 'value']
    varInfo = get_var_values(m)
    # print(varInfo)

    # Write to csv
    with open('model_out.csv', 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(columns)
        for varinfo in varInfo:
            wr.writerows(varinfo)

    return varInfo


def get_var_values(m: Model):
    variable_list = ['x_', 'y_', 's_', 'b_', 'dwwd', 'd2wwd']
    result = [[] for _ in range(len(variable_list))]
    for v in m.getVars():
        match = False
        for i in range(len(variable_list)):
            if match:
                break
            if variable_list[i] in v.varName and v.X > 0:
                match = True
                temp = v.varName.split('_')
                temp = [int(i) for i in temp[1:]]
                if i < 2:
                    result[i].append(temp + [v.varName, v.X])
                elif i == 2:
                    result[i].append(temp + ['-', v.varName, v.X])
                else:
                    result[i].append(temp + ['-', '-', v.varName, v.X])
    return result


def run_model(case, config):
    """
    Creates ModelData, ModelVars, Model instances to define and solve the model
    :param case_db: scenarios database
    :param config:  run configurations
    :return:
        model instance
    """
    random.seed(0)
    # Declare and initialize model
    m = Model('NFP')
    data = ModelData(case, config)
    time_horizon = data.get_last_elem(data.time_limit)
    plot_network(data.arcs_dep, data.distances, time_horizon)
    v = ModelVars()

    add_variables(m, data, v)
    add_constraints(m, data, v)
    add_symmetry_breaking_constr(m, data, v)
    add_objective(m, data, v)

    # Some model preferences to setup
    # m.setParam('Heuristics', 0.5)
    # m.setParam('MIPFocus', 1)
    # m.setParam('Threads', 12)
    # m.setParam('MIPGap', 0.1)
    m.setParam('Timelimit', 1000)
    # m.setParam('SolutionLimit', 1)
    m.update()

    # save the defined model in .lp format
    m.write('nfp.lp')
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        print('Optimal objective: %g' % m.ObjVal)
        # save the solution output
        m.write('nfp.sol')
        # write a csv file
        results = result_csv(m)
        arc2driver = get_driver_route(results, int(m.getObjective().getValue()))
        plot_network(arc2driver, data.distances, time_horizon, solved=True)
        return m
    elif m.Status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.Status)
        return m

    m.computeIIS()
    m.write('inf.ilp')
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


def plot_network(arcs_list, dist, time_horizon, solved=False):
    """
    Arc network plotting function. Shows the generated Arcs set on the timeline.
    :return: None
    """
    def arc_plot_iterator(arcs):
        for a in arcs:
            ax.plot([a[0], a[1]], [a[2], (a[2] + dist[min(a[0], a[1])]) % time_horizon], color)
    ax = plt.axes()
    if solved:
        for arcs in arcs_list:
            color = '#%06X' % random.randint(0, 0xFFFFFF)
            arc_plot_iterator(arcs)
    else:
        color = 'blue'
        arc_plot_iterator(arcs_list)
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Time (hours)')
    plt.show()


def get_driver_route(results, driver_count):
    driver_num = [i for i in range(driver_count)]
    print(driver_num)
    xy_arcs = results[0] + results[1]
    print(xy_arcs)
    result = [[] for _ in driver_num]
    for d in driver_num:
        for elem in xy_arcs:
            if elem[0] == d:
                result[d].append(elem[1:4])
    print(result)
    return result
