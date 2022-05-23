from gurobipy import tuplelist, tupledict


class ModelData:
    def __init__(self, case_db, config):
        """
        ModelData --- class for data processing. It's used in model definition.
        :param case_db: scenarios database
        :param config: run configurations
        """

        # catch the case run parameters
        self.case_id = config['scenario_number']
        self.cycle_length = config['cycle_length']

        # calculation of time horizon length corresponding to cycle length
        self.n_weeks = self.cycle_length // 7

        if self.n_weeks > 1:
            self.week_num = tuplelist(range(self.n_weeks))
            self.time_limit = tuplelist(
                [((i + 1) / self.n_weeks) * 24 * self.cycle_length for i in self.week_num])
            self.time_horizon = self.time_limit[-1]
        else:
            self.week_num = 0
            self.time_limit = 24 * self.cycle_length
            self.time_horizon = 24 * self.cycle_length

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

        # generate drivers set D
        self.drivers = tuplelist(d for d in range(0, 5 * self.n)) if self.n >= 3 else \
            tuplelist(d for d in range(0, 4 * self.n ** 2))  # set of drivers

        # catch forward/backward departure data
        self.departures = [self.cell_reader(case_db, 'Выезды прямо'),
                           self.cell_reader(case_db, 'Выезды обратно')]
        print('departures', self.departures)

        # generate forward/backward Arc matrix with departure and arriving info
        self.arcs_dep, self.arcs_arr = self.arcs_network_creator()  # set of arcs (works) to be served
        # print('arcs_arr', self.arcs_arr)

        # crew size for each arc
        self.c_a = arc_param(self.arcs_dep, self.crew_size)

        # arcs service durations
        self.t_a = arc_param(self.arcs_dep, self.distances)

        # unique time set T
        uniq_time_set = set([item[2] for item in self.arcs_dep] + [item[2] for item in self.arcs_arr])
        self.t_set = tuplelist(sorted(uniq_time_set))
        # print('t_set', self.t_set)

        # A_a_x and A_a_y set
        self.Aax = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 11, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with daily rest
        self.Aay = tupledict(
            {(i, j, t): find_closest_arrive((i, j, t), self.arcs_arr, self.distances, 24, self.time_horizon)
             for (i, j, t) in
             self.arcs_dep})  # set of arcs with the closest arrival time to departure arc a with weekly rest

        '''
        self.Aax_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 11), self.time_horizon)
            for (i, j, t) in self.arcs_dep})
        self.Aay_inv = tupledict({
            (i, j, t): find_closest_depart((i, j, t), self.arcs_dep, (self.t_a[i, j, t] + 24), self.time_horizon)
            for (i, j, t) in self.arcs_dep})
        '''

        self.Akw = self.arcs_week_subset(week='single')
        self.Akww = self.arcs_week_subset()  # set of arcs, which belongs to the double week 𝑘

        self.Ax = tupledict(
            {(i, t): find_closest_arrive((i, 0, t), self.arcs_arr, self.distances, 11, self.time_horizon)
             for i in
             self.nodes for t in
             self.t_set})  # set of arcs with the closest arrival time to departure arc a with daily rest
        # print(self.Ax)

        self.Ay = tupledict(
            {(i, t): find_closest_arrive((i, 0, t), self.arcs_arr, self.distances, 24, self.time_horizon)
             for i in
             self.nodes for t in
             self.t_set})  # set of arcs with the closest arrival time to departure arc a with weekly rest
        # print(self.Ay)

        self.is_arcs = tupledict(
            {(i, t): 1 if sum(self.t_a[ik, jk, tk] for (ik, jk, tk) in self.arcs_dep if
                              ik == i and tk == t) > 0 else 0 for (i, t) in self.Ax})

    def arcs_week_subset(self, week='single'):
        """
        get arc service time according to the week
        :param week: rule of arcs subset definition
        :return: set of arcs, which belongs to the week 𝑘 (𝑘 =[0, 1] for 'single' week, k=0 for 'double')
        """
        result = {}
        if week == 'single' and self.n_weeks > 1:
            for k in self.week_num:
                for (i, j, t) in self.arcs_dep:
                    if t < self.time_limit[k] and (k == 0 or self.time_limit[k - 1] < t):
                        result[k, i, j, t] = (self.time_limit[k] - t
                                              if t + self.t_a[i, j, t] > self.time_limit[k] else self.t_a[i, j, t])
                        if t + self.t_a[i, j, t] > self.time_limit[k]:
                            result[self.week_num[k - 1], i, j, t] = t + self.t_a[i, j, t] - self.time_limit[k]
        else:
            result = {(0, i, j, t): self.t_a[i, j, t] for (i, j, t) in self.arcs_dep}
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
        if isinstance(self.departures[0], list) and isinstance(self.departures[1], list):
            for cur_deps in zip(self.departures[0], self.departures[1]):
                temp = route_sim(cur_deps, self.distances, self.cycle_length)
                arcs_dep += temp[0]
                arcs_arr += temp[1]
        else:
            arcs_dep, arcs_arr = route_sim(self.departures, self.distances, self.cycle_length)
        # arcs_dep = sorted(arcs_dep, key=lambda item: item[2])
        # print(arcs_dep)
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
        return result if isinstance(result, int) else tuplelist(result)

    '''
    @staticmethod
    def get_last_elem(parameter):
        """
        returns last element of list or number, if input is integer
        :param parameter:
        :return:
        """
        return parameter if isinstance(parameter, int) else parameter[-1]

    '''

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


def arc_param(arcs, param):
    return tupledict({(i, j, t): param[min(i, j)] for (i, j, t) in arcs})


def route_sim(departures, distances, cycle_len):
    dep_forward = []
    dep_backward = []
    n = len(distances)
    time_limit = 24 * cycle_len
    arr_forward = []
    arr_backward = []
    for i in range(cycle_len):
        dep_forward_time = departures[0] + i * 24
        dep_backward_time = departures[1] + i * 24
        dep_forward.append([0, 1, dep_forward_time % time_limit])
        dep_backward.append([n, n - 1, dep_backward_time % time_limit])
        arr_forward.append([0, 1, (dep_forward_time + distances[0]) % time_limit])
        arr_backward.append([n, n - 1, (dep_backward_time + distances[-1]) % time_limit])
        for j in range(1, n):
            dep_forward_time += distances[j - 1]
            dep_backward_time += distances[n - j]
            dep_forward.append([j, j + 1, dep_forward_time % time_limit])
            dep_backward.append([n - j, n - j - 1, dep_backward_time % time_limit])
            arr_forward.append([j, j + 1, (dep_forward_time + distances[j]) % time_limit])
            arr_backward.append([n - j, n - j - 1, (dep_backward_time + distances[n - j - 1]) % time_limit])
    return dep_forward + dep_backward, arr_forward + arr_backward


def find_closest_arrive(a_, arcs_arr, arc_len, rest_time, time_limit):  # 11 or 24 relax time duration
    result = []
    time = a_[2] - rest_time
    t_closest = 2 * time_limit
    for a in arcs_arr[::-1]:
        if a[1] == a_[0]:
            if a[2] <= time:
                t_between = time - a[2]
            else:
                t_between = time - a[2] + time_limit
            if t_between <= t_closest:
                arc_dep_time = (a[2] - arc_len[min(a[0], a[1])]) if a[2] >= arc_len[min(a[0], a[1])] else \
                    (a[2] - arc_len[min(a[0], a[1])] + time_limit)
                if t_between < t_closest:
                    t_closest = t_between
                    result = [[a[0], a[1], arc_dep_time]]
                else:
                    result.append([a[0], a[1], arc_dep_time])

    # print('rel_time', rest_time, 'ans', a_, '==', result)
    return result


def find_closest_depart(a_, arcs_dep, rest_time, time_limit):  # 11 or 24 relax time duration
    result = []
    time = a_[2] + rest_time
    t_closest = 2 * time_limit
    for a in arcs_dep:
        if a[0] == a_[1]:
            if a[2] >= time:
                t_between = a[2] - time
            else:
                t_between = a[2] - time + time_limit
            if t_between <= t_closest:
                if t_between < t_closest:
                    t_closest = t_between
                    result = [a]
                else:
                    result.append(a)
    # print('rel_time', rest_time, 'ans', a_, '==', result)
    return result
