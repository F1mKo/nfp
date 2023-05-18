import os
from time import strftime, localtime
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, tuplelist, tupledict

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14,
                     'font.family': 'Times New Roman',
                     'mathtext.fontset': 'cm'})


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


def parse_data(input_file, sheet_name):
    with open(input_file):
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        df.set_index('ID', inplace=True)
        return df


class ModelData:
    def __init__(self, case_db, config):
        """
        ModelData --- class for data processing. It's used in model definition.
        :param case_db: scenarios database
        :param config: run configurations
        """

        # catch the case run parameters
        self.case_id = config['scenario_number']
        self.n_weeks = config['n_weeks'] + 1

        # calculation of time horizon length corresponding to cycle length
        self.cycle_length = 7 * self.n_weeks

        self.week_num = tuplelist(range(self.n_weeks))
        self.time_limit = tuplelist(
            [int(((i + 1) / self.n_weeks) * 24 * self.cycle_length) for i in self.week_num])
        self.time_horizon = self.time_limit[-1]

        print('n_weeks', self.n_weeks)
        print('time_limit', self.time_limit)

        # catch distances between nodes i and i+1
        self.distances = self.cell_reader(case_db, 'Участки')
        print('dist', self.distances)

        # calculation of total road fragments amount
        self.n = len(self.distances)

        # catch crew size values
        self.crew_size = self.cell_reader(case_db, 'Водители')
        print('crew_size', self.crew_size)

        # generate nodes set N
        self.nodes = tuplelist(i for i in range(self.n + 1))  # set of nodes in the network

        # catch forward/backward departure data
        self.departures = [self.cell_reader(case_db, 'Выезды прямо'),
                           self.cell_reader(case_db, 'Выезды обратно')]
        print('departures', self.departures)

        # generate drivers set D
        self.drivers = tuplelist(d for d in range(0, 7 * max([2, self.n]) * len(self.departures[0]) * (1 + sum([1 for i in self.crew_size if i == 2]))))  # set of drivers

        # generate forward/backward Arc matrix with departure and arriving info
        self.arcs_dep, self.arcs_arr = self.arcs_network_creator()  # set of arcs (works) to be served

        # crew size for each arc
        self.c_a = self.arc_param(self.arcs_dep, self.crew_size)

        # arcs service durations
        self.t_a = self.arc_param(self.arcs_dep, self.distances)

        # unique time set T
        uniq_time_set = set([item[2] for item in self.arcs_dep])
        self.t_set = tuplelist(sorted(uniq_time_set))
        print('t_set', self.t_set)

        self.possible_arc = calc_possible_arcs(self.nodes, self.arcs_dep)

        self.Aax = tupledict(
            {(i, j, t): find_closest_arrive_mod((i, j, t), self.possible_arc, self.distances, 11, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        self.Aay = tupledict(
            {(i, j, t): find_closest_arrive_mod((i, j, t), self.possible_arc, self.distances, 24, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest

        self.Akw = self.arcs_week_subset()

    def arcs_week_subset(self):
        """
        get arc service time according to the week
        :param week: rule of arcs subset definition
        :return: set of arcs, which belongs to the week 𝑘 (𝑘 =[0, 1] for 'single' week, 𝑘 =[0, 1, 2] for 'double')
        """
        result = {}
        for k in self.week_num:
            for (i, j, t) in self.arcs_dep:
                if t < self.time_limit[k] and (k == 0 or self.time_limit[k - 1] < t):
                    result[k, i, j, t] = (self.time_limit[k] - t
                                          if t + self.t_a[i, j, t] > self.time_limit[k] else self.t_a[i, j, t])
                    if t + self.t_a[i, j, t] > self.time_limit[k]:
                        result[self.week_num[k - 1], i, j, t] = t + self.t_a[i, j, t] - self.time_limit[k]
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
        for cur_deps in zip(self.departures[0], self.departures[1]):
            temp = route_sim(cur_deps, self.distances, self.n_weeks)
            arcs_dep += temp[0]
            arcs_arr += temp[1]
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
        result = self.split_data(case_db.loc[[self.case_id], cell_name].values[0])
        return tuplelist([result,]) if isinstance(result, int) else tuplelist(result)

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
            return int(data)
        else:
            return [int(i) for i in data.split(';')]

    @staticmethod
    def arc_param(arcs, parameter):
        """
        Simple map function to setup parameter values
        :param arcs: set of arcs to be served
        :param parameter: set of values corresponding to each node
        :return: parameter values corresponding to each arc
        """
        return tupledict({(i, j, t): parameter[min(i, j)] for (i, j, t) in arcs})


def route_sim(departures, distances, n_weeks):
    """
    Generation of arcs forward/ backward sets to be served
    :param departures: time of departure from route ends
    :param distances: time length corresponding to arc
    :param n_weeks: number of weeks to be generated
    :return: arrays of arcs to be served
    """
    dep_forward = []
    dep_backward = []
    n = len(distances)
    time_limit = 24 * 7
    arr_forward = []
    arr_backward = []
    for k in range(n_weeks):
        for i in range(7):
            dep_forward_time = departures[0] + i * 24
            dep_backward_time = departures[1] + i * 24
            dep_forward.append([0, 1, dep_forward_time % time_limit + time_limit * k])
            dep_backward.append([n, n - 1, dep_backward_time % time_limit + time_limit * k])
            arr_forward.append([0, 1, (dep_forward_time + distances[0]) % + time_limit * k])
            arr_backward.append([n, n - 1, (dep_backward_time + distances[-1]) % time_limit + time_limit * k])
            for j in range(1, n):
                dep_forward_time += distances[j - 1]
                dep_backward_time += distances[n - j]
                dep_forward.append([j, j + 1, dep_forward_time % time_limit + time_limit * k])
                dep_backward.append([n - j, n - j - 1, dep_backward_time % time_limit + time_limit * k])
                arr_forward.append([j, j + 1, (dep_forward_time + distances[j]) % time_limit + time_limit * k])
                arr_backward.append([n - j, n - j - 1, (dep_backward_time + distances[n - j - 1]) % time_limit + time_limit * k])
    return dep_forward + dep_backward, arr_forward + arr_backward


def calc_possible_arcs(nodes, arcs_dep):
    """
    sort possible to serve arc for each node
    :param nodes: nodes
    :param arcs_dep: set of arcs to be served
    :return: array of possible arc sets corresponding to each node
    """
    possible_route = [[] for node in nodes]
    for node in nodes:
        for a in arcs_dep:
            if a[1] == node:
                possible_route[node].append(a)
    #            arcs_dep.remove(a)
    return possible_route


def find_closest_arrive_mod(a_dep, possible_arc, arc_len, rest_time, time_horizon):  # 11 or 24 relax time duration
    """
    ***Cycled version*** Returns the closest arcs set to departure of arc a_ with taking into account rest time
    :param a_dep: given arc
    :param possible_arc: set of arcs to be selected as the closest
    :param arc_len: set of distances related to arc
    :param rest_time: rest time before departure on given arc a_
    :param time_horizon: time horizon
    :return:
    """
    result = []
    time = a_dep[2]
    t_closest = 2 * time_horizon
    for a in possible_arc[a_dep[0]]:
        arrival_time = a[2] + arc_len[min(a[0], a[1])] + rest_time
        if arrival_time <= time:
            t_between = time - arrival_time
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
        else:
            continue # comment it to make the function output to be time-cycled
            t_between = time - arrival_time + time_horizon
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
    return result


def find_closest_arrive(a_, arcs_arr, arc_len, rest_time, time_horizon):  # 11 or 24 relax time duration
    """
    ***Cycled version*** Returns the closest arcs set to departure of arc a_ with taking into account rest time
    :param a_: given arc
    :param arcs_arr: set of arcs to be selected as the closest
    :param arc_len: set of distances related to arc
    :param rest_time: rest time before departure on given arc a_
    :param time_horizon: time horizon
    :return:
    """
    result = []
    time = a_[2] - rest_time
    t_closest = 2 * time_horizon
    for a in arcs_arr[::-1]:
        if a[1] == a_[0]:
            if a[2] <= time:
                t_between = time - a[2]
            else:
                t_between = time - a[2] + time_horizon
            if t_between <= t_closest:
                arc_dep_time = (a[2] - arc_len[min(a[0], a[1])]) if a[2] >= arc_len[min(a[0], a[1])] else \
                    (a[2] - arc_len[min(a[0], a[1])] + time_horizon)
                if t_between < t_closest:
                    t_closest = t_between
                    result = [[a[0], a[1], arc_dep_time]]
                else:
                    result.append([a[0], a[1], arc_dep_time])
    return result


def plot_network(data, scenario_path):
    """
    Arc network plotting function. Shows the generated Arcs grid or optimal driver routes set on the timeline.
    :param data: model data
    :param scenario_path: folder path corresponding to case number to plot and files export
    :return: None
    """
    color = {-1: 'red', 1: 'blue', 0: 'gray'}

    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    # plt.title('Транспортная сеть - сценарий ' + scenario_path.split("/")[1])
    for a in data.arcs_dep:
        if a[2] + data.distances[min(a[0], a[1])] <= data.time_horizon:
            ax.plot([a[0], a[1]], [a[2], (a[2] + data.distances[min(a[0], a[1])])], color[a[0] - a[1]])
        else:
            if a[0] > a[1]:
                part_route_node = min(a[0], a[1]) + (max(a[0], a[1]) - min(a[0], a[1])) * \
                                  (1 - (data.time_horizon - a[2]) / data.distances[min(a[0], a[1])])
            else:
                part_route_node = max(a[0], a[1]) - (max(a[0], a[1]) - min(a[0], a[1])) * \
                                  (1 - (data.time_horizon - a[2]) / data.distances[min(a[0], a[1])])
            ax.plot([a[0], part_route_node], [a[2], data.time_horizon], color[a[0] - a[1]])
            ax.plot([part_route_node, a[1]],
                    [0, (a[2] + data.distances[min(a[0], a[1])]) % data.time_horizon], color[a[0] - a[1]])
    ax.set_xlabel('Точки смены экипажа ' + r'$(N)$')
    ax.set_ylabel('Время, час')
    plt.xlim([0 - 0.05 * len(data.distances), len(data.distances) + 0.05 * len(data.distances)])
    plt.xticks(range(len(data.distances)+1))
    plt.ylim([-1, data.time_horizon + 1])
    plt.yticks(range(0, data.time_horizon + 1, 24))
    plt.savefig(scenario_path + "/pictures/nfp_pic.pdf", format="pdf")
    plt.show()


def gantt_diagram(data, result_dict, scenario_path, true_numbers = False):
    """
    Drivers Gantt diagram plotting function. Shows the optimal driver schedule on the timeline.
    :param data: model data
    :param scenario_path: folder path corresponding to case number to plot and files export
    :return: None
    """

    color = {-1: 'red', 1: 'blue', 0: 'gray'}
    barch_hatch = {-1: '//', 1: '\\\\', 0: ''}

    hired_drivers = []
    for b in result_dict['b'].keys():
        if result_dict['b'][b] == 1:
            hired_drivers.append(b)

    for (i, d) in enumerate(hired_drivers):
        fig, gnt = plt.subplots(figsize=(7, max(4, len(data.distances) + 1)))
        gnt.grid(True)
        plt.title("Маршрут водителя {0}".format(d if true_numbers else i))
        gnt.set_xlim(0, data.time_horizon)
        gnt.set_ylim(0 - 1 / 4, 1 * len(data.distances) + 1 / 4)
        gnt.set_xlabel('Время, час')
        gnt.set_ylabel('Точки смены экипажа ' + r'$(N)$')
        gnt.set_xticks(range(0, data.time_horizon + 1, 24))
        gnt.set_yticks([i * 1 for i in range(len(data.distances) + 1)])
        gnt.set_xticklabels(range(0, data.time_horizon + 1, 24))
        gnt.set_yticklabels(range(len(data.distances) + 1))

        for arc_key in data.arcs_dep:
            if result_dict['x'][tuple([d] + arc_key)] or result_dict['y'][tuple([d] + arc_key)] == 1:
                if arc_key[2] + data.distances[min(arc_key[0], arc_key[1])] <= data.time_horizon:
                    gnt.broken_barh([(arc_key[2], data.distances[min(arc_key[0], arc_key[1])])],
                                    (min(arc_key[0], arc_key[1]) * 1 + 1/4, 1/2),
                                    facecolors=color[arc_key[0]-arc_key[1]], hatch=barch_hatch[arc_key[0]-arc_key[1]])
                else:
                    gnt.broken_barh([(arc_key[2], data.time_horizon - arc_key[2])],
                                    (min(arc_key[0], arc_key[1]) * 1 + 1/4, 1/2),
                                    facecolors=color[arc_key[0]-arc_key[1]], hatch=barch_hatch[arc_key[0]-arc_key[1]])

                    gnt.broken_barh([(0, (arc_key[2] + data.distances[min(arc_key[0], arc_key[1])]) % data.time_horizon)],
                                    (min(arc_key[0], arc_key[1]) * 1 + 1/4, 1/2),
                                    facecolors=color[arc_key[0]-arc_key[1]], hatch=barch_hatch[arc_key[0]-arc_key[1]])
        for node_key in data.nodes:
            for time in data.t_set:
                if result_dict['s'][tuple([d, node_key, time])] == 1:
                    if time == data.t_set[-1]:
                        gnt.broken_barh([(time, data.time_horizon - time)],
                                        (node_key * 1 - 1 / 5, 2 / 5), facecolors=color[0], hatch=barch_hatch[0])
                        gnt.broken_barh([(0, data.t_set[0])],
                                        (node_key * 1 - 1 / 5, 2 / 5), facecolors=color[0], hatch=barch_hatch[0])
                    else:
                        tk = sum(data.t_set[k + 1] for k in range(len(data.t_set))
                                 if data.t_set[k] == time and k < len(data.t_set) - 1)
                        gnt.broken_barh([(time, tk - time)],
                                        (node_key * 1 - 1 / 5, 2 / 5), facecolors=color[0], hatch=barch_hatch[0])
        plt.savefig(scenario_path + "/pictures/driver_{0}_route.pdf".format(d if true_numbers else i), format="pdf", bbox_inches='tight')
        plt.show(bbox_inches='tight')


def get_var_values(m: Model):
    """
    Get variables from the optimal solution
    :param m: model to extract solution data
    :return: optimal solution data
    """
    all_vars = m.getVars()
    var_names = []
    var_set = {}
    for v in all_vars:
        v_full_name = v.varName
        if '(' in v_full_name:
            v_name, v_domains = v_full_name.split('(')
            v_domains = v_domains[:-1].split(',')
            var_names.append(v_name)
            if var_set.get(v_name) is None:
                var_set[v_name] = len(v_domains)
        else:
            var_names.append(v_full_name)
            if var_set.get(v_full_name) is None:
                var_set[v_full_name] = 0

    var_dict = {var: pd.DataFrame() for var in var_set.keys()}
    for v in all_vars:
        v_full_name = v.varName
        if '(' in v_full_name:
            v_name, v_domains = v_full_name.split('(')
            v_domains = [int(s) if s.isdigit() else s for s in v_domains[:-1].split(',')]
            if var_dict.get(v_name) is not None:
                var_dict[v_name] = pd.concat([var_dict[v_name],
                                              pd.Series(v_domains + [v.x, v.lb, v.ub]).to_frame().T],
                                             axis=0,
                                             ignore_index=True)
        else:
            if var_dict.get(v_full_name) is not None:
                var_dict[v_full_name] = pd.concat([var_dict[v_full_name],
                                                   pd.Series([v.x, v.lb, v.ub]).to_frame().T],
                                                  axis=0,
                                                  ignore_index=True)

    var_dict_records = {}
    for var in var_set.keys():
        if var_set[var] > 0:
            var_dict[var].sort_values(by=list(var_dict[var].columns)[:var_set[var]], inplace=True)

        var_dict[var].columns = ['Column' + str(i + 1) for i in range(var_set[var])] + ['Value', 'Lower bound',
                                                                                        'Upper bound']
        records = var_dict[var].loc[:, :'Value'].astype(int)
        if var_set[var] > 0:
            var_dict_records[var] = records.set_index(['Column' + str(i + 1) for
                                                       i in range(var_set[var])]).to_dict()['Value']
        else:
            var_dict_records[var] = records['Value'].values

    var_dict['objective'] = pd.DataFrame(data=[m.objval], columns=['Value'])

    return var_dict, var_dict_records


def sol2excel(result, run_path, db: dict = None, folder_name: str = None):
    # Creating Excel Writer Object from Pandas
    with pd.ExcelWriter(run_path + '/' + (folder_name if folder_name is not None else '') +
                        '_result.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        if db is not None:
            for param in db.keys():
                worksheet = workbook.add_worksheet(param)
                writer.sheets[param] = worksheet
                db[param].to_excel(writer, sheet_name=param, startrow=0, startcol=0, index=False)

                # Get the dimensions of the dataframe.
                (max_row, max_col) = db[param].shape

                # Make the columns wider for clarity.
                worksheet.set_column(0, max_col - 1, 20)

                # Set the auto-filter.
                worksheet.autofilter(0, 0, max_row, max_col - 1)

        for variable in result.keys():
            worksheet = workbook.add_worksheet(variable)
            writer.sheets[variable] = worksheet
            result[variable].to_excel(writer, sheet_name=variable, startrow=0, startcol=0, index=False)

            # Get the dimensions of the dataframe.
            (max_row, max_col) = result[variable].shape

            # Make the columns wider for clarity.
            worksheet.set_column(0, max_col - 1, 20)

            # Set the auto-filter.
            worksheet.autofilter(0, 0, max_row, max_col - 1)
