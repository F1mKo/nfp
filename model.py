from gurobipy import Model, tupledict, GRB, quicksum
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
    v.w_work_d = tupledict(
        {(k, d): m.addVar(vtype=GRB.CONTINUOUS, name="dwwd_{0}_{1}".format(k, d)) for d in
         data.drivers for k in data.week_num})

    v.ww_work_d = tupledict(
        {d: m.addVar(vtype=GRB.CONTINUOUS, name="d2wwd_{0}".format(d)) for d in
         data.drivers})


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
                                             + quicksum(v.y_da[d, i2, j2, t2] for (i2, j2, t2) in data.Aay[i, j, t]),
                                             name="driver_movement_{0}_{1}_{2}_{3}".format(d, i, j, t))
                             for d in data.drivers for (i, j, t) in data.arcs_dep})


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
    driver_wwd_def = tupledict({d: m.addConstr(
        quicksum(data.Akw[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akw if
                 k == ki) == v.w_work_d[ki, d], name="driver_w_wd_definition_{0}_{1}".format(ki, d))
        for d in data.drivers for ki in data.week_num})

    driver_2wwd_def = tupledict({d: m.addConstr(
        quicksum(data.Akww[k, i, j, t] * (v.x_da[d, i, j, t] + v.y_da[d, i, j, t]) for (k, i, j, t) in data.Akww) ==
        v.ww_work_d[d], name="driver_2w_wd_definition_{0}".format(d))
        for d in data.drivers})

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
        v.x_da.sum(d, '*', '*', '*') + v.y_da.sum(d, '*', '*', '*') <= (2 * data.n * data.cycle_length) * v.b_d[d],
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
    add_symmetry_breaking_constr(m, data, v)
    add_objective(m, data, v)
