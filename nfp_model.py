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


def add_variables(m: Model, data: ModelData, v: ModelVars, start_node=True):
    """
    Defines variables in model according to data.
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :param start_node: choose model variable start_d node version if True else alternative arcs version
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
    if start_node:
        v.start_d = tupledict({(d, i, t): m.addVar(vtype=GRB.BINARY,
                                                   name="start_{0}_{1}_{2}".format(d, i, t))
                               for d in data.drivers for i in data.nodes for t in data.t_set if t == data.t_set[0]})
    else:
        v.start_d = tupledict({(d, i, j, t): m.addVar(vtype=GRB.BINARY,
                                                      name="start_{0}_{1}_{2}_{3}".format(d, i, j, t))
                               for d in data.drivers for (i, j, t) in data.arcs_dep if t == data.t_set[0]})

    # driver single/double week work duration
    v.w_work_d = tupledict(
        {(d, k): m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}_{1}".format(d, k)) for d in
         data.drivers for k in data.week_num})

    v.ww_work_d = tupledict(
        {(d, k2[0], k2[1]): m.addVar(vtype=GRB.CONTINUOUS, name="d2wwd_{0}_{1}_{2}".format(d, k2[0], k2[1])) for d in
         data.drivers for k2 in combinations(data.week_num, 2) if (abs(k2[1] - k2[0]) == 1)})


def add_driver_movement(m: Model, data: ModelData, v: ModelVars, start_node=True):
    """
    Define main model constraints block
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :param start_node: choose model variable start_d node version if True else alternative arcs version
    :return: None
    """
    if start_node:
        tupledict({(d, n, ts): m.addConstr(
            quicksum((quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[n, j, ts])
                      + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[n, j, t]))
                     for (i, j, t) in data.arcs_dep if
                     ((j == n + 1) or (j == n - 1)) and i == n and t == ts) + quicksum(
                v.s_dit[d, n, data.t_set[k - 1]] for k in range(len(data.t_set))
                if data.t_set[k] == ts) == v.s_dit[d, n, ts] + v.x_da.sum(d, n, '*', ts)
            + v.y_da.sum(d, n, '*', ts) - (v.start_d[d, n, ts] if ts == data.t_set[0] else 0),
            name="driver_movement_{0}_{1}_{2}".format(d, n, ts))
            for d in data.drivers for n in data.nodes for ts in data.t_set})

        tupledict({d: m.addConstr(v.start_d.sum(d, '*', '*') == v.b_d[d], name="driver_start_definition_{0}".format(d))
                   for d in data.drivers})
    else:
        tupledict({(d, i, j, t): m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] ==
                                             quicksum(v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == t)
                                             + quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[i, j, t])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]) +
                                             (v.start_d[d, i, j, t] if t == data.t_set[0] else 0),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                   for d in data.drivers for (i, j, t) in data.arcs_dep})
        '''
        d_move_main = tupledict({(d, n, ts):
            m.addConstr(
                quicksum((quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[n, j, ts])
                          + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[n, j, t]))
                         for (i, j, t) in data.arcs_dep if
                         ((j == n + 1) or (j == n - 1)) and i == n and t == ts) +
                quicksum(v.s_dit[d, n, data.t_set[k - 1]] for k in range(len(data.t_set))
                         if data.t_set[k] == ts and ts != data.t_set[0])
                == v.s_dit[d, n, ts] - (v.start_d.sum(d, n, '*', ts) if ts == data.t_set[0] else 0)
                + v.x_da.sum(d, n, '*', ts) + v.y_da.sum(d, n, '*', ts),
                name="driver_movement_{0}_{1}_{2}".format(d, n, ts))
            for d in data.drivers for n in data.nodes for ts in data.t_set})
        '''
        tupledict({d: m.addConstr(
            v.start_d.sum(d, '*', '*', '*') == v.b_d[d],
            name="driver_start_definition_{0}".format(d))
            for d in data.drivers})

    tupledict({(d, i, t): m.addConstr(v.x_da.sum(d, i, '*', t) + v.y_da.sum(d, i, '*', t) + v.s_dit.sum(d, '*', t)
                                      <= v.b_d[d], name="driver_movement_idle_deny_{0}_{1}_{2}".format(d, i, t))
               for d in data.drivers for i in data.nodes for t in data.t_set})


def non_useful_function():
    """
    Keep different constraint versions code
    """
    '''
    d_move_main = tupledict({(d, i, j, t):
                                 m.addConstr(v.s_dit[d, i, t] + v.x_da[d, i, j, t] + v.y_da[d, i, j, t] ==
                                             quicksum(v.s_dit[d, i, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                      if data.t_set[k] == t)
                                             + quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[i, j, t])
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]) +
                                             (v.start_d[d, i, j, t] if t == data.t_set[0] else 0),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                             for d in data.drivers for (i, j, t) in data.arcs_dep})

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
                                 m.addConstr(
                                     quicksum((quicksum(v.x_da[d, i1, j1, t1] for (i1, j1, t1) in data.Aax[n, j, ts])
                                               + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[n, j, t]))
                                              for (i, j, t) in data.arcs_dep if
                                              ((j == n + 1) or (j == n - 1)) and i == n and t == ts) +
                                         quicksum(v.s_dit[d, n, data.t_set[k - 1]] for k in range(len(data.t_set))
                                                  if data.t_set[k] == ts and ts != data.t_set[0])
                                     == v.s_dit[d, n, ts] - (v.start_d.sum(d, n, '*', ts) if ts == data.t_set[0] else 0)
                                     + v.x_da.sum(d, n, '*', ts) + v.y_da.sum(d, n, '*', ts),
                                     name="driver_movement_{0}_{1}_{2}".format(d, n, ts))
                             for d in data.drivers for n in data.nodes for ts in data.t_set})

    d_start_constr = tupledict({d: m.addConstr(
        v.start_d.sum(d, '*', '*', '*') == v.b_d[d],
        name="driver_start_definition_{0}".format(d))
        for d in data.drivers})

    d_start_constr = tupledict({d: m.addConstr(
        v.start_d.sum(d, '*', '*') == v.b_d[d],
        name="driver_start_definition_{0}".format(d))
        for d in data.drivers})
    '''


def add_week_work_constraints(m: Model, data: ModelData, v: ModelVars):
    """
    Defines a constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """

    # Create crew size constraints
    tupledict({(i, j, t): m.addConstr(quicksum(v.x_da[d, i, j, t] + v.y_da[d, i, j, t] for d in data.drivers) ==
                                      data.c_a[i, j, t], name="crew_size_constr_{0}_{1}_{2}".format(i, j, t))
               for (i, j, t) in data.arcs_dep})

    # Driver week work time definition and constraints
    tupledict({(d, k1): m.addConstr(quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t])
                                             for (k, i, j, t) in data.Akw if k == k1) == v.w_work_d[d, k1],
                                    name="driver_w_wd_definition_{0}_{1}".format(d, k1))
               for d in data.drivers for k1 in data.week_num})

    tupledict({(d, k2[0], k2[1]): m.addConstr(quicksum(quicksum(
        data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if k == k1)
                                                       for k1 in k2) ==
                                              v.ww_work_d[d, k2[0], k2[1]],
                                              name="driver_2w_wd_definition_{0}_{1}_{2}".format(d, k2[0], k2[1]))
               for d in data.drivers for k2 in combinations(data.week_num, 2) if (abs(k2[1] - k2[0]) == 1)})

    tupledict({(d, k): m.addConstr(v.w_work_d[d, k] <= 56, name="driver_w_wd_constraints_{0}_{1}".format(d, k))
               for (d, k) in v.w_work_d})

    tupledict({(d, k2[0], k2[1]): m.addConstr(v.ww_work_d[d, k2[0], k2[1]] <= 90,
                                              name="driver_2w_wd_constraints_{0}_{1}_{2}".format(d, k2[0], k2[1]))
               for d in data.drivers for k2 in combinations(data.week_num, 2) if
               (abs(k2[1] - k2[0]) == 1)})

    # Create weekly equality constraints
    tupledict({(d, i, j, t, k1): (m.addConstr(v.x_da[d, i, j, t] == v.x_da[d, i, j, (t + 168 * k1) % data.time_horizon],
                                              name="weekly_equality_constraints_x_{0}_{1}_{2}_{3}".format(d, i, j, t,
                                                                                                          k1)))
               for (i, j, t) in data.arcs_dep
               for d in data.drivers for k1 in data.week_num[1:] if t <= 168})

    tupledict({(d, i, j, t, k1): (m.addConstr(v.y_da[d, i, j, t] == v.y_da[d, i, j, (t + 168 * k1) % data.time_horizon],
                                              name="weekly_equality_constraints_y_{0}_{1}_{2}_{3}".format(d, i, j, t,
                                                                                                          k1)))
               for (i, j, t) in data.arcs_dep
               for d in data.drivers for k1 in data.week_num[1:] if t <= 168})

    # Create weekly rest constraints
    tupledict({(d, ki): m.addConstr(quicksum(v.y_da[d, i, j, t] for (k, i, j, t) in data.Akw if k == ki) >= v.b_d[d],
                                    name="weekly_rest_constraints_{0}_{1}".format(d, ki)) for d in
               data.drivers for ki in data.week_num})


def add_symmetry_breaking_constr(m: Model, data: ModelData, v: ModelVars):
    """
    Defines an additional symmetry breaking constraints block for model according to data
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    # Create driver work_time symmetry breaking constraints
    tupledict({data.drivers[i]: m.addConstr(v.w_work_d[0, data.drivers[i + 1]] <= v.w_work_d[0, data.drivers[i]],
                                            name="symmetry_breaking_wwd_constraints_{0}".format(
                                                data.drivers[i])) for i in range(len(data.drivers) - 1)})
    # Create driver selection symmetry breaking constraints
    tupledict({data.drivers[i]: m.addConstr(v.b_d[data.drivers[i]] >= v.b_d[data.drivers[i + 1]],
                                            name="symmetry_breaking_ds_constraints_{0}".format(
                                                data.drivers[i])) for i in range(len(data.drivers) - 1)})


def add_objective(m: Model, data: ModelData, v: ModelVars):
    """
    Defines an objective function for model
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    #   Create driver selection definition
    tupledict({d: m.addConstr(
        v.x_da.sum(d, '*', '*', '*') + v.y_da.sum(d, '*', '*', '*') <= (data.n * data.cycle_length) * v.b_d[d],
        name="driver_selection_definition_{0}".format(d))
        for d in data.drivers})

    m.setObjective(v.b_d.sum('*'), GRB.MINIMIZE)


def fix_arcs(v: ModelVars, solution):
    """
    fix arcs related to initial solution
    :param v:
    :param solution:
    :return:
    """
    for a in solution:
        if 'x' in a[4]:
            [d, i, j, t] = [int(i) for i in a[:4]]
            v.x_da[d, i, j, t].lb = 1
        elif 'y' in a[4]:
            [d, i, j, t] = [int(i) for i in a[:4]]
            v.y_da[d, i, j, t].lb = 1


def constraint_creator(m: Model, data: ModelData, v: ModelVars, start_node=True):
    """
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :param start_node: choose model variable start_d node version if True else alternative arcs version
    :return: None
    """
    add_driver_movement(m, data, v, start_node)
    add_week_work_constraints(m, data, v)
    # add_symmetry_breaking_constr(m, data, v)
    add_objective(m, data, v)
