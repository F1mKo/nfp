from datetime import datetime
import pandas as pd
from gurobipy import Model, GRB
from model_data import ModelData, plot_network, result_csv, get_driver_route, read_sol_csv
from model import ModelVars, add_variables, add_driver_movement_basic, add_driver_movement_alt_logic, add_week_work_constraints, \
    add_symmetry_breaking_constr, add_objective, constraint_creator, fix_arcs
import random

config = {"input_file": "scenarios.xlsx",
          'sheet_name': 'augmentation',
          'scenario_number': '10737_1',
          'cycle_length': 7}  # settings

# 10372_1
# 10737_1 # it works a bit on alternative version
# 10733_1 # 6 nodes dimension
# 30748_1 not work (array of departs)
# 12372_1 not work (one distance processing error)


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df


if __name__ == '__main__':
    now = datetime.now()
    case = parse_data(config['input_file'], config['sheet_name'])
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
    v = ModelVars()

    plot_network(data.arcs_dep, data.distances, data.t_set, data.time_horizon)

    add_variables(m, data, v)
    # constraint_creator(m, data, v, baseline=True)
    constraint_creator(m, data, v, baseline=False)

    # start_sol = read_sol_csv()
    # print(start_sol)
    # fix_arcs(m, data, v, solution=start_sol)

    # Some model preferences to setup
    # m.setParam('Heuristics', 0.5)
    m.setParam('MIPFocus', 1)
    m.setParam('Threads', 12)
    # m.setParam('MIPGap', 0.1)
    m.setParam('Timelimit', 3000)
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
        arc2driver, node2driver = get_driver_route(results, int(m.getObjective().getValue()))
        plot_network(arc2driver, data.distances, data.t_set, data.time_horizon, solved=True, idle_nodes=node2driver)
    elif m.Status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.Status)
    else:
        m.computeIIS()
        m.write('inf.ilp')

    print('Total execution time', datetime.now() - now)


