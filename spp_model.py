from gurobipy import Model, tupledict, GRB, quicksum
from model_data import ModelData
from itertools import combinations


class SPPModelVars:
    def __init__(self):
        """
        ModelVars --- class for variable definition. It stores all variables for convenient use in model.
        """
        self.x_i = tupledict()  # binary variable, equals to 1 if duty 𝑖 ∈ 𝐷 is selected as optimal, 0 otherwise


def add_variables(m: Model, data: ModelData, v: SPPModelVars, start_node = True):
    """
    Defines variables in model according to data.
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :return: None
    """
    v.x_i = tupledict({i: m.addVar(vtype=GRB.BINARY, name="x_{0}".format(i)) for i in data.D})


def model_creator(m: Model, data: ModelData, v: SPPModelVars):
    """
    :param m: Model class instance
    :param data: ModelData class instance
    :param v: ModelVars class instance
    :param baseline: chose base model version if True else alternative constraints version
    :return: None
    """
    duty_selection = tupledict({j: m.addConstr(
            quicksum(v.x_i[i] for i in data.D if data.Ai[i, j] == 1) == v.c_j[j],
            name="arc_crew_size_check_{0}".format(j))
            for j in data.A})
    #   Objective definition
    m.setObjective(v.x_i.sum('*'), GRB.MINIMIZE)
