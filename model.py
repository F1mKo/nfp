from gurobipy import Model, tupledict, GRB, quicksum
from model_data import ModelData
from itertools import combinations


class ModelVars:
    def __init__(self):
        """
        ModelVars --- class for variable definition. It stores all variables for convenient use in model.
        """
        self.x_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴, 0 otherwise
        self.y_da = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 serves arc 𝑎 ∈ 𝐴 and have a weekly rest on the end node of arc, 0 otherwise
        self.s_dit = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is located in node 𝑖 ∈ 𝑁 at time 𝑡, 0 otherwise
        self.b_d = tupledict()  # binary variable, equals to 1 if driver 𝑑 ∈ 𝐷 is selected, 0 otherwise
        self.start_d = tupledict()
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

    v.start_d = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                                  name="start_{0}_{1}_{2}_{3}".format(d, i, j, t))
                           for d in data.drivers for (i, j, t) in data.arcs_dep if t == data.t_set[0]})
    #v.start_d = tupledict({(d, i, t): m.addVar(vtype=GRB.BINARY,
    #                                              name="start_{0}_{1}_{2}".format(d, i, t))
    #                       for d in data.drivers for i in data.nodes for t in data.t_set if t == data.t_set[0]})

    # driver single/double week work duration
    v.w_work_d = tupledict(
        {(d, k): m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}_{1}".format(d, k)) for d in
         data.drivers for k in data.week_num})

    v.ww_work_d = tupledict(
        {(d, k2[0], k2[1]): m.addVar(vtype=GRB.CONTINUOUS, name="d2wwd_{0}_{1}_{2}".format(d, k2[0], k2[1])) for d in
         data.drivers for k2 in combinations(data.week_num, 2) if (abs(k2[1] - k2[0]) == 1)})


def add_driver_movement_basic(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data

    Base version of main logic and constraints

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
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]) +
                                             (v.start_d[d, i, j, t] if t == data.t_set[0] else 0),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                             for d in data.drivers for (i, j, t) in data.arcs_dep})
    '''
    d_move_main = tupledict({(d, n, ts):
                                 m.addConstr(quicksum((quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[n, j, ts])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[n, j, t]))
                                                      for (i, j, t) in data.arcs_dep if ((j == n + 1) or (j == n - 1)) and i == n and t == ts) + quicksum(v.s_dit[d, n, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == ts)
                                             == v.s_dit[d, n, ts]
                                              + v.x_da.sum(d, n, '*', ts) + v.y_da.sum(d, n, '*', ts) - (v.start_d[d, n, ts] if ts == data.t_set[0] else 0),
                                             name="driver_movement_{0}_{1}_{2}".format(d, n, ts))
                             for d in data.drivers for n in data.nodes for ts in data.t_set})

    d_move_main = tupledict({(d, n, ts):
                                 m.addConstr(quicksum((quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[n, j, ts])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[n, j, t]))
                                                      for (i, j, t) in data.arcs_dep if ((j == n + 1) or (j == n - 1)) and i == n and t == ts) + quicksum(v.s_dit[d, n, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == ts)
                                             == v.s_dit[d, n, ts]
                                              + v.x_da.sum(d, n, '*', ts) + v.y_da.sum(d, n, '*', ts) - (v.start_d.sum(d, n, '*', '*') if ts == data.t_set[0] else 0),
                                             name="driver_movement_{0}_{1}_{2}".format(d, n, ts))
                             for d in data.drivers for n in data.nodes for ts in data.t_set})
    '''

    d_start_constr = tupledict({d: m.addConstr(
        v.start_d.sum(d, '*', '*', '*') == v.b_d[d],
        name="driver_start_definition_{0}".format(d))
        for d in data.drivers})


    #d_start_constr = tupledict({d: m.addConstr(
    #    v.start_d.sum(d, '*', '*') == v.b_d[d],
    #    name="driver_start_definition_{0}".format(d))
    #    for d in data.drivers})

    #   Create driver idle node definition

    d_idle_constr = tupledict({(d, t): m.addConstr(
        v.s_dit.sum(d, '*', t) <= v.b_d[d],
        name="driver_idle_constraints_{0}_{0}".format(d, t))
        for d in data.drivers for t in data.t_set})

    d_move_no_idle = tupledict({(d, i, t):
        m.addConstr(
            v.x_da.sum(d, i, '*', t) + v.y_da.sum(d, i, '*', t) <= 1 - v.s_dit[d, i, t],
            name="driver_movement_idle_deny_{0}_{1}_{2}".format(d, i, t))
        for d in data.drivers for i in data.nodes for t in data.t_set})



def add_driver_movement_alt_logic(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data

    Alternative version of model constraints

    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """

    # Driver movement definition
    '''
    d_move_alt = tupledict({(d, i, t):
                                m.addConstr(v.s_dit[d, i, t] + v.x_da.sum(d, i, '*', t) +
                                            v.y_da.sum(d, i, '*', t) == quicksum(
                                    v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                    if data.t_set[k] == t) + quicksum(v.x_da[d, ik, jk, tk] +
                                            v.y_da[d, ik, jk, tk] for (ik, jk, tk) in data.arcs_dep if jk == i and
                                                                     tk == t - data.distances[min(ik, jk)])
                                           ,
                                            name="driver_movement_{0}_{1}_{2}".format(d, i, t))
                            for d in data.drivers for i in data.nodes for t in data.t_set})
    '''

    d_move_alt = tupledict({(d, i, t):
                                m.addConstr(v.s_dit[d, i, t] + v.x_da.sum(d, i, '*', t) +
                                            v.y_da.sum(d, i, '*', t) == quicksum(
                                    v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                    if data.t_set[k] == t)
                                            + quicksum(
                                    v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Ax[i, t] if data.is_arcs[i, t] == 1
                                ) +
                                            quicksum(
                                                v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Ay[i, t] if
                                                data.is_arcs[i, t] == 1),
                                            name="driver_movement_{0}_{1}_{2}".format(d, i, t))
                            for d in data.drivers for i in data.nodes for t in data.t_set})
    '''                       
    d_move_alt = tupledict({(d, i, t):
                                m.addConstr(v.s_dit[d, i, t] + v.x_da.sum(d, i, '*', t) +
                                                                   v.y_da.sum(d, i, '*', t) == quicksum(
                                                v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                if data.t_set[k] == t),
                                            name="driver_movement_{0}_{1}_{2}".format(d, i, t))
                            for d in data.drivers for i in data.nodes for t in data.t_set})

    '''
    '''
    d_move_add00 = tupledict({(d, i, j, t):
                                  m.addConstr(v.s_dit[d, j, (t + data.distances[
                                  min(i, j)]) % data.time_horizon] >= v.x_da[d, i, j, t],
                                              name="driver_movement_rest_x_{0}_{1}_{2}_{3}".format(d, i, j, t))
                              for d in data.drivers for (i, j, t) in data.arcs_dep})

    d_move_add01 = tupledict({(d, i, j, t):
                                  m.addConstr(v.s_dit[d, j, (t + data.distances[
                                  min(i, j)]) % data.time_horizon] >= v.y_da[d, i, j, t],
                                              name="driver_movement_rest_y_{0}_{1}_{2}_{3}".format(d, i, j, t))
                              for d in data.drivers for (i, j, t) in data.arcs_dep})
    '''
    d_move_add1 = tupledict({(d, i, j, t):
        m.addConstr(
            quicksum(v.x_da[d, ik, jk, tk] + v.y_da[d, ik, jk, tk] for (ik, jk, tk) in
                     data.arcs_dep if
                     (t + data.distances[min(i, j)] + 11 > data.time_horizon) and (
                             tk <= data.time_horizon and
                             (jk != j and t <= tk or (ik == i and jk == j) and t < tk) or 0 <= tk < (
                                     t + data.distances[min(i, j)] + 11) % data.time_horizon)
                     or
                     (t + data.distances[min(i, j)] + 11 <= data.time_horizon) and (tk < t + data.distances[
                         min(i, j)] + 11) and
                     (jk != j and t <= tk or (ik == i and jk == j) and t < tk))
            <= 1 - v.x_da[d, i, j, t],
            name="driver_movement3_{0}_{1}_{2}_{3}".format(d, i, j, t))
        for d in data.drivers for (i, j, t) in data.arcs_dep})

    d_move_add2 = tupledict({(d, i, j, t):
        m.addConstr(
            quicksum(v.x_da[d, ik, jk, tk] + v.y_da[d, ik, jk, tk] for (ik, jk, tk) in
                     data.arcs_dep if
                     (t + data.distances[min(i, j)] + 24 > data.time_horizon) and (
                             tk <= data.time_horizon and
                             (jk != j and t <= tk or (ik == i and jk == j) and t < tk) or 0 <= tk < (
                                     t + data.distances[min(i, j)] + 24) % data.time_horizon)
                     or
                     (t + data.distances[min(i, j)] + 24 <= data.time_horizon) and (tk < t + data.distances[
                         min(i, j)] + 24) and
                     (jk != j and t <= tk or (ik == i and jk == j) and t < tk))
            <= 1 - v.y_da[d, i, j, t],
            name="driver_movement4_{0}_{1}_{2}_{3}".format(d, i, j, t))
        for d in data.drivers for (i, j, t) in data.arcs_dep})

    #   Create driver idle node definition
    d_idle_constr = tupledict({(d, t): m.addConstr(
        v.s_dit.sum(d, '*', t) <= v.b_d[d],
        name="driver_idle_constraints_{0}_{0}".format(d, t))
        for d in data.drivers for t in data.t_set})

    d_move_no_idle = tupledict({(d, i, t):
        m.addConstr(
            v.x_da.sum(d, i, '*', t) + v.y_da.sum(d, i, '*', t) <= 1 - v.s_dit[d, i, t],
            name="driver_movement_idle_deny_{0}_{1}_{2}".format(d, i, t))
        for d in data.drivers for i in data.nodes for t in data.t_set})


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

    # Driver week work time definition and constraints
    driver_wwd_def = tupledict({(d, k1): m.addConstr(
        quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if
                 k == k1) == v.w_work_d[d, k1], name="driver_w_wd_definition_{0}_{1}".format(d, k1))
        for d in data.drivers for k1 in data.week_num})

    driver_2wwd_def = tupledict({(d, k2[0], k2[1]): m.addConstr(
        quicksum(quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if k == k1) for k1 in k2) ==
        v.ww_work_d[d, k2[0], k2[1]], name="driver_2w_wd_definition_{0}_{1}_{2}".format(d, k2[0], k2[1]))
        for d in data.drivers for k2 in combinations(data.week_num, 2) if (abs(k2[1] - k2[0]) == 1)})

    driver_wwd_constr = tupledict({(d, k): m.addConstr(v.w_work_d[d, k] <= 56,
                                                       name="driver_w_wd_constraints_{0}_{1}".format(d, k))
                                   for (d, k) in v.w_work_d})

    driver_2wwd_constr = tupledict({(d, k2[0], k2[1]): m.addConstr(v.ww_work_d[d, k2[0], k2[1]] <= 90,
                                                   name="driver_2w_wd_constraints_{0}_{1}_{2}".format(d, k2[0], k2[1]))
                                    for d in data.drivers for k2 in combinations(data.week_num, 2) if (abs(k2[1] - k2[0]) == 1)})

    # Create weekly rest constraints
    weekly_rest_constr = tupledict(
        {(d, ki): m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw if k == ki) >= v.b_d[d],
                        name="weekly_rest_constraints_{0}_{1}".format(d, ki)) for d in
         data.drivers for ki in data.week_num})

    # Create weekly equality constraints
    weekly_equality_constr_x = tupledict(
        {(d, i, j, t, k1): (m.addConstr(v.x_da[d, i, j, t] == v.x_da[d, i, j, (t + 168 * k1) % data.time_horizon],
                        name="weekly_equality_constraints_x_{0}_{1}_{2}_{3}".format(d, i, j, t, k1))) for (i, j, t) in data.arcs_dep
             for d in data.drivers for k1 in data.week_num[1:] if t <= 168})

    weekly_equality_constr_y = tupledict(
        {(d, i, j, t, k1): (m.addConstr(v.y_da[d, i, j, t] == v.y_da[d, i, j, (t + 168 * k1) % data.time_horizon],
                        name="weekly_equality_constraints_y_{0}_{1}_{2}_{3}".format(d, i, j, t, k1))) for (i, j, t) in data.arcs_dep
             for d in data.drivers for k1 in data.week_num[1:] if t <= 168})


def add_symmetry_breaking_constr(m: Model, data: ModelData, v: ModelVars):
    """
    Defines an additional symmetry breaking constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    # Create driver work_time symmetry breaking constraints
    symmetry_breaking_wwd_constraints = {
        data.drivers[i]: m.addConstr(v.w_work_d[0, data.drivers[i + 1]] <= v.w_work_d[0, data.drivers[i]],
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
    """
    Defines an objective function for model
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    #   Create driver selection definition
    d_selection_def = tupledict({d: m.addConstr(
        v.x_da.sum(d, '*', '*', '*') + v.y_da.sum(d, '*', '*', '*') <= (100 * data.n * data.cycle_length) * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})

    m.setObjective(v.b_d.sum('*'), GRB.MINIMIZE)


def fix_arcs(m: Model, data: ModelData, v: ModelVars, solution):
    for a in solution:
        if 'x' in a[4]:
            [d, i, j, t] = [int(i) for i in a[:4]]
            v.x_da[d, i, j, t].lb = 1
        elif 'y' in a[4]:
            [d, i, j, t] = [int(i) for i in a[:4]]
            v.y_da[d, i, j, t].lb = 1


def constraint_creator(m: Model, data: ModelData, v: ModelVars, baseline=True):
    """
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :param baseline: chose base model version if True else alternative constraints version
    :return: None
    """
    if baseline:
        add_driver_movement_basic(m, data, v)
    else:
        add_driver_movement_alt_logic(m, data, v)

    add_week_work_constraints(m, data, v)
    # add_symmetry_breaking_constr(m, data, v)
    add_objective(m, data, v)
