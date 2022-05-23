from gurobipy import Model, tuplelist, tupledict, GRB, quicksum
import matplotlib.pyplot as plt
import csv
import random
from model_data import ModelData


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
            {d: m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}".format(d)) for d in
             data.drivers})


def add_driver_movement_basic(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    # Base version

    # Driver movement definition
    d_move_main = tupledict({(d, i, j, t):
                                 m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] ==
                                             quicksum(v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == t)
                                             + quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[i, j, t])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                             for d in data.drivers for (i, j, t) in data.arcs_dep})

    #   Create driver selection definition
    d_selection_def = tupledict({d: m.addConstr(
        quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for (i, j, t) in data.arcs_dep) <= 10000 * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})


def add_driver_movement_logic(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """

    # Driver movement definition
    d_move_main = tupledict({(d, i, j, t):
                                 m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] ==
                                             quicksum(v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == t)
                                             + quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[i, j, t])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                             for d in data.drivers for (i, j, t) in data.arcs_dep})

    d_move1 = tupledict({(d, i, t):
                             m.addConstr(v.s_dit[d, i, t] + sum(v.x_da.select(d, i, '*', t) +
                                                                v.y_da.select(d, i, '*', t)) ==
                                         quicksum(
                                             v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                             if
                                             data.t_set[k] == t)
                                         + quicksum(
                                 v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Ax[i, t] if data.is_arcs[i, t] == 1
                             ) +
                                         quicksum(
                                             v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Ay[i, t] if
                                             data.is_arcs[i, t] == 1),
                                         name="driver_movement_{0}_{1}_{2}".format(d, i, t))
                         for d in data.drivers for i in data.nodes for t in data.t_set})


    d_move_add00 = tupledict({(d, i, j, t):
                                  m.addConstr(v.s_dit[d, j, tk] >= v.x_da[d, i, j, t],
                                              name="driver_movement_rest_x_{0}_{1}_{2}_{3}".format(d, i, j, t))
                              for d in data.drivers for (i, j, t) in data.arcs_dep for tk in data.t_set if
                              ((t + data.distances[
                                  min(i, j)] <= tk < t + data.distances[min(i, j)] + 11)
                               or (0 <= tk < (t + data.distances[min(i, j)] + 11) % data.time_horizon) and
                               (t + data.distances[
                                   min(i, j)] + 11 >= data.time_horizon))})

    d_move_add01 = tupledict({(d, i, j, t):
                                  m.addConstr(v.s_dit[d, j, tk] >= v.y_da[d, i, j, t],
                                              name="driver_movement_rest_y_{0}_{1}_{2}_{3}".format(d, i, j, t))
                              for d in data.drivers for (i, j, t) in data.arcs_dep for tk in data.t_set if
                              ((t + data.distances[
                                  min(i, j)] <= tk < t + data.distances[min(i, j)] + 24)
                               or (0 <= tk < (t + data.distances[min(i, j)] + 24) % data.time_horizon) and
                               (t + data.distances[
                                   min(i, j)] + 24 >= data.time_horizon))})


    d_move_no_idle = tupledict({(d, i, j, t):
                                  m.addConstr(v.s_dit[d, i, t] <= 1 - v.x_da[d, i, j, t] - v.y_da[d, i, j, t],
                                              name="driver_movement_idle_deny_{0}_{1}_{2}_{3}".format(d, i, j, t))
                              for d in data.drivers for (i, j, t) in data.arcs_dep})

    d_move_add0 = tupledict({(d, i, t):
                                 m.addConstr(quicksum(v.x_da[d, ik, jk, tk] +
                                                      v.y_da[d, ik, jk, tk] for (ik, jk, tk) in
                                                      data.arcs_dep if
                                                      ik == i and tk == t) <= 1,
                                             name="driver_movement2_{0}_{1}_{2}".format(d, i, t))
                             for d in data.drivers for i in data.nodes for t in data.t_set})


    d_move_add1 = tupledict({(d, i, j, t):
        m.addConstr(
            quicksum(v.x_da[d, ik, jk, tk] + v.y_da[d, ik, jk, tk] for (ik, jk, tk) in
                     data.arcs_dep if
                     ik == j and ((t < tk < t + data.distances[
                         min(i, j)] + 11 <= data.time_horizon)
                                  or (
                                          t + data.time_horizon < tk + data.time_horizon < t +
                                          data.distances[
                                              min(i, j)] + 11 + data.time_horizon and
                                          t + data.distances[
                                              min(i, j)] + 11 > data.time_horizon)) or
                     ik != j and ((t <= tk < t + data.distances[
                         min(i, j)] + 11 <= data.time_horizon)
                                  or (
                                          t + data.time_horizon <= tk + data.time_horizon < t +
                                          data.distances[
                                              min(i, j)] + 11 + data.time_horizon and
                                          t + data.distances[
                                              min(i, j)] + 11 > data.time_horizon))
                     )
            <= 1 - v.x_da[d, i, j, t],
            name="driver_movement3_{0}_{1}_{2}_{3}".format(d, i, j, t))
        for d in data.drivers for (i, j, t) in data.arcs_dep})

    d_move_add2 = tupledict({(d, i, j, t):
        m.addConstr(
            quicksum(v.x_da[d, ik, jk, tk] + v.y_da[d, ik, jk, tk] for (ik, jk, tk) in
                     data.arcs_dep if
                     ik == j and ((t < tk < t + data.distances[
                         min(i, j)] + 24 <= data.time_horizon)
                                  or (
                                          t + data.time_horizon < tk + data.time_horizon < t +
                                          data.distances[
                                              min(i, j)] + 24 + data.time_horizon and
                                          t + data.distances[
                                              min(i, j)] + 24 > data.time_horizon)) or
                     ik != j and ((t <= tk < t + data.distances[
                         min(i, j)] + 24 <= data.time_horizon)
                                  or (
                                          t + data.time_horizon <= tk + data.time_horizon < t +
                                          data.distances[
                                              min(i, j)] + 24 + data.time_horizon and
                                          t + data.distances[
                                              min(i, j)] + 24 > data.time_horizon)))
            <= 1 - v.y_da[d, i, j, t],
            name="driver_movement4_{0}_{1}_{2}_{3}".format(d, i, j, t))
        for d in data.drivers for (i, j, t) in data.arcs_dep})

    #   Create driver selection definition
    d_selection_def = tupledict({d: m.addConstr(
        quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for (i, j, t) in data.arcs_dep) <= 10000 * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})

    #   Create driver idle node definition
    d_idle_constr = tupledict({(d, t): m.addConstr(
        v.s_dit.sum(d, '*', t) <= v.b_d[d],
        name="driver_idle_constraints_{0}_{0}".format(d, t))
        for d in data.drivers for t in data.t_set})


def add_week_work_constraints(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """

    # Create crew size constraints
    crew_size_constr = tupledict({
        (i, j, t): m.addConstr(
            quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for d in data.drivers) == data.c_a[i, j, t],
            name="crew_size_constr_{0}_{1}_{2}".format(i, j, t))
        for (i, j, t) in data.arcs_dep})

    if data.n_weeks > 1:
        # Driver week work time definition and constraints
        driver_wwd_def = tupledict({d: m.addConstr(
            quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if
                     k == ki) == v.w_work_d[ki, d], name="driver_w_wd_definition_{0}_{1}".format(ki, d))
            for d in data.drivers for ki in data.week_num})

        driver_2wwd_def = tupledict({d: m.addConstr(
            quicksum(data.Akww[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akww) ==
            v.ww_work_d[d], name="driver_2w_wd_definition_{0}".format(d))
            for d in data.drivers for ki in data.week_num})

        driver_wwd_constr = tupledict({(k, d): m.addConstr(v.w_work_d[k, d] <= 56,
                                                           name="driver_w_wd_constraints_{0}_{1}".format(k, d))
                                       for (k, d) in v.w_work_d})

        driver_2wwd_constr = tupledict({d: m.addConstr(v.ww_work_d[d] <= 90,
                                                       name="driver_2w_wd_constraints_{0}".format(d))
                                        for d in data.drivers})

        # Create weekly rest constraints
        weekly_rest_constr = tupledict(
            {d: m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw if k == ki) >= v.b_d[d],
                            name="weekly_rest_constraints_{0}".format(d)) for d in
             data.drivers for ki in data.week_num})
    else:
        # Driver week work time definition and constraints
        driver_wwd_def = tupledict({d: m.addConstr(
            quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw) ==
            v.w_work_d[d], name="driver_w_wd_definition_{0}".format(d))
            for d in data.drivers})

        driver_wwd_constr = tupledict({d: m.addConstr(v.w_work_d[d] <= 56,
                                                      name="driver_w_wd_constraints_{0}".format(d))
                                       for d in data.drivers})
        # Create weekly rest constraints
        weekly_rest_constr = tupledict(
            {d: m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw) >= v.b_d[d],
                            name="weekly_rest_constraints_{0}".format(d)) for d in
             data.drivers})


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
    m.setObjective(v.b_d.sum('*'), GRB.MINIMIZE)


def result_csv(m: Model):
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
                    result[i].append(temp[:-1] + ['-', temp[-1], v.varName, v.X])
                else:
                    result[i].append(temp + ['-', '-', v.varName, v.X])
    return result


def plot_network(arcs_list, dist, t_set, time_horizon, solved=False, idle_nodes=None):
    """
    Arc network plotting function. Shows the generated Arcs set on the timeline.
    :return: None
    """

    def plot_iterator(arcs, is_idles=False):
        if not is_idles:
            for a in arcs:
                # ax.plot([a[0], a[1]], [a[2], (a[2] + dist[min(a[0], a[1])]) % time_horizon], color)
                ax.plot([a[0], a[1]], [a[2], (a[2] + dist[min(a[0], a[1])])], color)
        else:
            for i in arcs:
                # print(i)
                if i[2] == t_set[-1]:
                    ax.plot([i[1], i[1]], [i[2], time_horizon], color)
                    ax.plot([i[1], i[1]], [0, t_set[0]], color)
                else:
                    tk = sum(t_set[k + 1] for k in range(len(t_set))
                             if t_set[k] == i[2] and k < len(t_set) - 1)
                    # print(tk)
                    ax.plot([i[1], i[1]], [i[2], tk], color)

    if solved:
        d = 0
        for arcs, idle in zip(arcs_list, idle_nodes):
            plt.figure()
            ax = plt.axes()
            plt.title("driver_{0}_route".format(d))
            color = '#%06X' % random.randint(0, 0xFFFFFF)
            plot_iterator(arcs)
            plot_iterator(idle, is_idles=True)
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Time (hours)')
            plt.show()
            d += 1
            # break
    else:
        ax = plt.axes()
        color = 'blue'
        plot_iterator(arcs_list)
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Time (hours)')
        plt.show()


def get_driver_route(results, driver_count):
    driver_num = [i for i in range(driver_count)]
    # print(driver_num)
    xy_arcs = results[0] + results[1]
    # print(xy_arcs)
    result = [[] for _ in driver_num]
    idles = [[] for _ in driver_num]
    for d in driver_num:
        for elem in xy_arcs:
            if elem[0] == d:
                result[d].append(elem[1:4])

        for elem in results[2]:
            if elem[0] == d:
                idles[d].append(elem[:2] + [elem[3]])
    # print(result)
    # print(idles)
    return result, idles
