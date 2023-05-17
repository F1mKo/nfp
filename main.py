from datetime import datetime
import os
import pandas as pd
from gurobipy import Model, GRB
from model_data import ModelData, plot_network, result_csv, get_driver_route, gantt_diagram
from nfp_model import ModelVars, add_variables, constraint_creator
import random
from time import gmtime, strftime, localtime


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df


def output_folder_check(scenario_folder):
    def check_path(path_to_check):
        # Check whether the specified path exists or not
        is_exist = os.path.exists(path_to_check)

        if not is_exist:
            # Create a new directory because it does not exist
            os.makedirs(path_to_check)

        return path_to_check

    def check_run_path(run_path, save_time=True):
        if save_time:
            new_path = run_path + '__' + ''.join(strftime("%Y_%m_%d_%H_%M_%S", localtime()))
            os.makedirs(new_path)
            return new_path
        else:
            # Check whether the specified path exists or not
            is_exist = os.path.exists(run_path)
            if not is_exist:
                os.makedirs(run_path)
            return run_path

    path = check_path('results')
    scenario_path = check_run_path(path + '/' + scenario_folder, save_time=True)
    check_path(scenario_path + '/pictures')

    return scenario_path


def run_opt(run_config, res_file):
    """
    Creates ModelData, ModelVars, Model instances to define and solve the model
    :param case_db: scenarios database
    :param config:  run configurations
    :return:
        model instance
    """
    print('RUN scenario ' + run_config['scenario_number'])

    now = datetime.now()
    scenario_path = output_folder_check(run_config['scenario_number'])
    case = parse_data(run_config['input_file'], run_config['sheet_name'])
    random.seed(0)
    # Declare and initialize model
    m = Model(run_config['model_type'])
    data = ModelData(case, run_config)
    v = ModelVars()
    start_node = True

    plot_network(data.arcs_dep, data.distances, data.t_set, data.time_horizon, scenario_path)

    add_variables(m, data, v, start_node)
    constraint_creator(m, data, v, start_node)

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
        res_file.write('Optimal objective for ' + run_config['scenario_number'] +': %g' % m.ObjVal)
        res_file.write('\n')
        # save the solution output
        if m.ObjVal < GRB.INFINITY:
            m.write(scenario_path + '/nfp.sol')
            # write a csv file
            results, hired_drivers = result_csv(m, scenario_path)
            arc2driver, node2driver = get_driver_route(results, hired_drivers)
            gantt_diagram(arc2driver, data.distances, data.t_set, data.time_horizon, scenario_path,
                          idle_nodes=node2driver, hired_drivers=hired_drivers)

    elif m.Status != GRB.INFEASIBLE:
        print(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write('\n')
    else:
        print(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write('\n')
        # m.computeIIS()
        # m.write(scenario_path + '/inf.ilp')

    print('Total execution time', datetime.now() - now)


if __name__ == '__main__':
    config = {'input_file': 'scenarios.xlsx',
              'sheet_name': 'augmentation',
              'model_type': 'NFP',
              'scenario_number': '10737_1',
              'n_weeks': 1}  # settings

    with open("run_results.txt", "a") as file:
        run_opt(config, file)



