from gurobipy import Model, tuplelist, tupledict
import matplotlib.pyplot as plt
import csv
import random

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14,
                     'font.family': 'Times New Roman',
                     'mathtext.fontset': 'cm'})
# plt.rcParams["figure.figsize"] = (10,3)

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
        # self.drivers = tuplelist(d for d in range(0, 16))  # set of drivers
        self.drivers = tuplelist(d for d in range(0, 7 * max([2, self.n]) * len(self.departures[0]) * (1 + sum([1 for i in self.crew_size if i == 2]))))  # set of drivers

        # generate forward/backward Arc matrix with departure and arriving info
        self.arcs_dep, self.arcs_arr = self.arcs_network_creator()  # set of arcs (works) to be served
        # print('arcs_arr', self.arcs_arr)

        # crew size for each arc
        self.c_a = self.arc_param(self.arcs_dep, self.crew_size)

        # arcs service durations
        self.t_a = self.arc_param(self.arcs_dep, self.distances)

        # unique time set T
        # uniq_time_set = set([item[2] for item in self.arcs_dep] + [item[2] for item in self.arcs_arr])
        uniq_time_set = set([item[2] for item in self.arcs_dep])
        self.t_set = tuplelist(sorted(uniq_time_set))
        print('t_set', self.t_set)

        self.possible_arc = calc_possible_arcs(self.nodes, self.arcs_dep)

        # A_a_x and A_a_y set
        #self.Aax = tupledict(
        #    {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 11, self.time_horizon)
        #     for (i, j, t) in
        #     self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        #self.Aay = tupledict(
        #    {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 24, self.time_horizon)
        #     for (i, j, t) in
        #     self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest

        self.Aax = tupledict(
            {(i, j, t): find_closest_arrive_mod((i, j, t), self.possible_arc, self.distances, 11, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        self.Aay = tupledict(
            {(i, j, t): find_closest_arrive_mod((i, j, t), self.possible_arc, self.distances, 24, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest

        self.Akw = self.arcs_week_subset()
        # self.Akww = self.arcs_week_subset()  # set of arcs, which belongs to the double week 𝑘

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


def result_csv(m: Model, scenario_path):
    """
    Catches variables values from model optimization results. Creates a csv-type file with determined columns
    :param m: Model class instance
    :return: csv-type file and array of hired driver numbers
    """
    columns = ['Driver', 'i', 'j', 'time', 'variable', 'value']
    varInfo = get_var_values(m)

    # Write to csv
    with open(scenario_path + '/model_out.csv', 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(columns)
        for varinfo in varInfo:
            wr.writerows(varinfo)

    hired_drivers = []
    for v in m.getVars():
        if 'b_' in v.varName and v.X > 0:
            temp = v.varName.split('_')
            temp = [int(i) for i in temp[1:]]
            hired_drivers.append(temp[-1])

    return varInfo, hired_drivers


def read_sol_csv(filename='10737_1.csv'):
    """
    Read existing scenario solution
    :param filename: name of csv file
    :return: optimal solution data
    """
    result = []
    with open(filename, newline='\n') as my_file:
        for line in csv.reader(my_file, delimiter=',', quotechar='"'):
            result.append(line)
    return result


def get_var_values(m: Model):
    """
    Get variables from the optimal solution
    :param m: model to extract solution data
    :return: optimal solution data
    """
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
                    result[i].append([temp[-1]] + ['-', '-', '-', v.varName, v.X])
    return result


def get_driver_route(results, driver_num):
    """
    Get driver optimal routes
    :param results: array with optimal solution data
    :param driver_num: array with hired driver numbers
    :return: drivers routes and idles
    """
    # print(driver_num)
    xy_arcs = results[0] + results[1]
    # print(xy_arcs)
    result = [[] for _ in driver_num]
    idles = [[] for _ in driver_num]
    for (i, d) in enumerate(driver_num):
        for elem in xy_arcs:
            if elem[0] == d:
                result[i].append(elem[1:4])
        for elem in results[2]:
            if elem[0] == d:
                idles[i].append(elem[:2] + [elem[3]])
    # print(result)
    # print(idles)
    return result, idles


def plot_network(arcs_list, dist, t_set, time_horizon, scenario_path, solved=False, idle_nodes=None, hired_drivers=None):
    """
    Arc network plotting function. Shows the generated Arcs grid or optimal driver routes set on the timeline.
    :param arcs_list: generated Arcs set or optimal driver arc subsets to serve
    :param dist: set of arc time durations
    :param t_set: time grid
    :param time_horizon: time horison
    :param scenario_path: folder path corresponding to case number to plot and files export
    :param solved: bool value to control plotting
    :param idle_nodes: driver idle timelines
    :param hired_drivers: list of hired drivers
    :return: None
    """

    def plot_iterator(arcs, is_idles=False):
        if not is_idles:
            for a in arcs:
                if a[2] + dist[min(a[0], a[1])] <= time_horizon:
                    ax.plot([a[0], a[1]], [a[2], (a[2] + dist[min(a[0], a[1])])], color)
                else:
                    if a[0] > a[1]:
                        part_route_node = min(a[0], a[1]) + (max(a[0], a[1]) - min(a[0], a[1])) * \
                                      (1 - (time_horizon - a[2]) / dist[min(a[0], a[1])])
                    else:
                        part_route_node = max(a[0], a[1]) - (max(a[0], a[1]) - min(a[0], a[1])) * \
                                          (1 - (time_horizon - a[2]) / dist[min(a[0], a[1])])
                    ax.plot([a[0], part_route_node], [a[2], time_horizon], color)
                    ax.plot([part_route_node, a[1]], [0, (a[2] + dist[min(a[0], a[1])]) % time_horizon], color)
        else:
            for i in arcs:
                if i[2] == t_set[-1]:
                    ax.plot([i[1], i[1]], [i[2], time_horizon], color)
                    ax.plot([i[1], i[1]], [0, t_set[0]], color)
                else:
                    tk = sum(t_set[k + 1] for k in range(len(t_set))
                             if t_set[k] == i[2] and k < len(t_set) - 1)
                    ax.plot([i[1], i[1]], [i[2], tk], color)

    if solved:
        d = 0
        for arcs, idle in zip(arcs_list, idle_nodes):
            plt.figure()
            ax = plt.axes()
            plt.title("Маршрут водителя {0}".format(hired_drivers[d]))
            color = '#%06X' % random.randint(0, 0xFFFFFF)
            plot_iterator(arcs)
            plot_iterator(idle, is_idles=True)
            ax.set_xlabel('Точки смены экипажа ' + r'$(N)$')
            ax.set_ylabel('Время, час')
            plt.xlim([0 - 0.05 * len(dist), len(dist) + 0.05 * len(dist)])
            plt.xticks(range(len(dist)+1))
            plt.ylim([-1, time_horizon + 1])
            plt.yticks(range(0, time_horizon + 1, 24))
            plt.savefig(scenario_path + "/pictures/driver_{0}_route.pdf".format(hired_drivers[d]), format="pdf")
            plt.show()
            d += 1
    else:
        plt.figure(figsize=(10, 8))
        ax = plt.axes()
        # plt.title('Транспортная сеть - сценарий ' + scenario_path.split("/")[1])
        color = 'blue'
        plot_iterator(arcs_list)
        ax.set_xlabel('Точки смены экипажа ' + r'$(N)$')
        ax.set_ylabel('Время, час')
        plt.xlim([0 - 0.05 * len(dist), len(dist) + 0.05 * len(dist)])
        plt.xticks(range(len(dist)+1))
        plt.ylim([-1, time_horizon + 1])
        plt.yticks(range(0, time_horizon + 1, 24))
        plt.savefig(scenario_path + "/pictures/nfp_pic.pdf", format="pdf")
        plt.show()


def gantt_diagram(arcs_list, dist, t_set, time_horizon, scenario_path, idle_nodes=None, hired_drivers=None, true_numbers = False):
    """
    Drivers Gantt diagram plotting function. Shows the optimal driver schedule on the timeline.
    :param arcs_list: generated Arcs set or optimal driver arc subsets to serve
    :param dist: set of arc time durations
    :param t_set: time grid
    :param time_horizon: time horison
    :param scenario_path: folder path corresponding to case number to plot and files export
    :param idle_nodes: driver idle timelines
    :param hired_drivers: list of hired drivers
    :return: None
    """
    def plot_iterator(arcs, is_idles=False):
        if not is_idles:
            for a in arcs:
                # ax.plot([a[0], a[1]], [a[2], (a[2] + dist[min(a[0], a[1])]) % time_horizon], color)
                if a[2] + dist[min(a[0], a[1])] <= time_horizon:
                    gnt.broken_barh([(a[2], dist[min(a[0], a[1])])], (min(a[0], a[1]) * 1 + 1/4, 1/2), facecolors=color[a[0]-a[1]])
                else:
                    gnt.broken_barh([(a[2], time_horizon - a[2])], (min(a[0], a[1]) * 1 + 1/4, 1/2), facecolors=color[a[0]-a[1]])
                    # gnt.plot([a[0], part_route_node], [a[2], time_horizon], color)
                    gnt.broken_barh([(0, (a[2] + dist[min(a[0], a[1])]) % time_horizon)], (min(a[0], a[1]) * 1 + 1/4, 1/2), facecolors=color[a[0]-a[1]])
                    # gnt.plot([part_route_node, a[1]], [0, (a[2] + dist[min(a[0], a[1])]) % time_horizon], color)
        else:
            for i in arcs:
                # print(i)
                if i[2] == t_set[-1]:
                    gnt.broken_barh([(i[2], time_horizon - i[2])],
                                    (i[1] * 1 - 1 / 5, 2 / 5), facecolors=color[0])
                    gnt.broken_barh([(0, t_set[0])],
                                    (i[1] * 1 - 1 / 5, 2 / 5), facecolors=color[0])
                else:
                    tk = sum(t_set[k + 1] for k in range(len(t_set))
                             if t_set[k] == i[2] and k < len(t_set) - 1)
                    gnt.broken_barh([(i[2], tk - i[2])],
                                    (i[1] * 1 - 1 / 5, 2 / 5), facecolors=color[0])

    d = 0
    for arcs, idle in zip(arcs_list, idle_nodes):
        fig, gnt = plt.subplots(figsize=(15, 6))
        gnt.grid(True)
        plt.title("Маршрут водителя {0}".format(hired_drivers[d] if true_numbers else d))
        gnt.set_xlim(-1, time_horizon + 1)
        gnt.set_ylim(0 - 1 / 2 , 1 * len(dist) + 1 / 2)
        gnt.set_xlabel('Время, час')
        gnt.set_ylabel('Точки смены экипажа ' + r'$(N)$')
        gnt.set_xticks(range(0, time_horizon + 1, 24))
        gnt.set_yticks([i * 1 for i in range(len(dist) + 1)])
        gnt.set_xticklabels(range(0, time_horizon + 1, 24))
        gnt.set_yticklabels(range(len(dist) + 1))

        color = {-1: 'red', 1: 'blue', 0: 'gray'}
        plot_iterator(arcs)
        plot_iterator(idle, is_idles=True)
        plt.savefig(scenario_path + "/pictures/driver_{0}_route.pdf".format(hired_drivers[d] if true_numbers else d), format="pdf")
        plt.show()
        d += 1
