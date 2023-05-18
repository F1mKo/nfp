from datetime import datetime
from gurobipy import Model, GRB
from model_data import output_folder_check, parse_data, ModelData, plot_network,\
    gantt_diagram, get_var_values, sol2excel
from nfp_model import ModelVars, add_variables, constraint_creator


def run_opt(run_config, res_file):
    """
    Creates ModelData, ModelVars, Model instances to define and solve the model
    :param run_config: run configurations
    :param res_file:  file to gather run status information
    :return:
        model instance
    """
    print('RUN scenario ' + run_config['scenario_number'])

    now = datetime.now()
    run_path = output_folder_check(run_config['scenario_number'])
    case = parse_data(run_config['input_file'], run_config['sheet_name'])
    # Declare and initialize model
    m = Model(run_config['model_type'])
    data = ModelData(case, run_config)
    v = ModelVars()
    start_node = True

    plot_network(data, run_path)

    add_variables(m, data, v, start_node)
    constraint_creator(m, data, v, start_node)

    # Some model preferences to setup
    # m.setParam('Heuristics', 0.5)
    m.setParam('MIPFocus', 1)
    m.setParam('Threads', 12)
    # m.setParam('MIPGap', 0.1)
    m.setParam('Timelimit', 14400)
    m.setParam('LogFile', run_path + '/m.log')
    # m.setParam('SolutionLimit', 1)
    m.update()

    # save the defined model in .lp format
    m.write(run_path + '/nfp.lp')
    m.optimize()

    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        print('Optimal objective: %g' % m.ObjVal)
        res_file.write('Optimal objective for ' + run_config['scenario_number'] +': %g' % m.ObjVal)
        res_file.write('\n')
        # save the solution output
        if m.ObjVal < GRB.INFINITY:
            m.write(run_path + '/nfp.sol')
            result_df, result_dict = get_var_values(m)
            print("Writing solution to excel:")
            sol2excel(result_df, run_path)
            gantt_diagram(data, result_dict, run_path, true_numbers=False)

    elif m.Status != GRB.INFEASIBLE:
        print(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write('\n')
    else:
        print(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write(run_config['scenario_number'] + ' : Optimization was stopped with status %d' % m.Status)
        res_file.write('\n')
        m.computeIIS()
        m.write(run_path + '/inf.ilp')

    print('Total execution time', datetime.now() - now)


if __name__ == '__main__':
    config = {'input_file': 'input_data.xlsx',
              'sheet_name': 'cases',
              'model_type': 'NFP',
              'scenario_number': 'hard',
              'n_weeks': 1}  # settings

    with open("run_results.txt", "a") as file:
        run_opt(config, file)



