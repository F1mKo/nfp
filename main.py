from datetime import datetime
import os
import pandas as pd
from gurobipy import Model, GRB
from model_data import ModelData, plot_network, result_csv, get_driver_route, gantt_diagram
from nfp_model import ModelVars, add_variables, constraint_creator
import random

config = {'input_file': 'scenarios.xlsx',
          'sheet_name': 'augmentation',
          'model_type': 'NFP',
          'scenario_number': '10733_1',
          'n_weeks': 1}  # settings


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df


def output_folder_check(scenario_folder):
    def check_path(path, rewrite_flag=True):
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            return path
        elif not rewrite_flag:
            new_path = path + '__' + ''.join(random.choice(['1','2','3','4','5','6','7','8','9']) for i in range(3))
            os.makedirs(new_path)
            return new_path
        else:
            return path

    path = check_path('results')
    scenario_path = check_path(path + '/' + scenario_folder, rewrite_flag=False)
    check_path(scenario_path + '/pictures')

    return scenario_path


if __name__ == '__main__':
    """
    Creates ModelData, ModelVars, Model instances to define and solve the model
    :param case_db: scenarios database
    :param config:  run configurations
    :return:
        model instance
    """
    now = datetime.now()
    scenario_path = output_folder_check(config['scenario_number'])
    case = parse_data(config['input_file'], config['sheet_name'])
    random.seed(0)
    # Declare and initialize model
    m = Model(config['model_type'])
    data = ModelData(case, config)
    v = ModelVars()
    start_node = True # если True, определяет переменную start_d как нахождение водителя в узле i в t_set[0], иначе start_d - ребро a с отправлением в t_set[0]

    plot_network(data.arcs_dep, data.distances, data.t_set, data.time_horizon, scenario_path)

    add_variables(m, data, v, start_node)
    constraint_creator(m, data, v, start_node)
    # constraint_creator(m, data, v, baseline=False)

    # fix previous solution to search infeasibility
    # start_sol = read_sol_csv()
    # print(start_sol)
    # fix_arcs(m, data, v, solution=start_sol)

    # Some model preferences to setup
    # m.setParam('Heuristics', 0.5)
    m.setParam('MIPFocus', 1)
    m.setParam('Threads', 12)
    # m.setParam('MIPGap', 0.1)
    m.setParam('Timelimit', 14400)
    m.setParam('LogFile', scenario_path + '/m.log')
    # m.setParam('SolutionLimit', 1)
    m.update()

    # save the defined model in .lp format
    m.write(scenario_path + '/nfp.lp')
    m.optimize()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        print('Optimal objective: %g' % m.ObjVal)
        # save the solution output
        m.write(scenario_path + '/nfp.sol')
        # write a csv file
        results, hired_drivers = result_csv(m, scenario_path)
        arc2driver, node2driver = get_driver_route(results, hired_drivers)
        # plot_network(arc2driver, data.distances, data.t_set, data.time_horizon, data.case_id,  solved=True, idle_nodes=node2driver, hired_drivers=hired_drivers)
        gantt_diagram(arc2driver, data.distances, data.t_set, data.time_horizon, scenario_path, idle_nodes=node2driver, hired_drivers=hired_drivers)
    elif m.Status != GRB.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.Status)
    else:
        m.computeIIS()
        m.write(scenario_path + '/inf.ilp')

    print('Total execution time', datetime.now() - now)


