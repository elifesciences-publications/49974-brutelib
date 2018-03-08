import logging as log
import os
from collections import deque

import dill as dill_pickle
import h5py
import numpy
import numpy as np
import scipy
import tables
from scipy.integrate import ode
from tqdm import tqdm


class SimulationState(object):
    def get_time(self):
        return self.t

    def get_discrete_system_state(self):
        return self.b

    def get_continuous_system_state(self):
        return self.y

    def get_random_state_generator(self):
        return self.rng

    def get_ode_instance(self):
        return self.ode

    def __init__(self, t, b, y, rng, ode):
        self.t = t
        self.b = b
        self.y = y
        self.rng = rng
        self.ode = ode


class AbstractMonitor(object):
    def record(self, t):
        """ decide whether a recording has to happen

        :param t: the current simulation time
        :return: is_record_time
        """
        return False

    def make_record(self, last_ts, last_bs, last_ys, simulation_state=None):
        """ Record some state-dependent values

        :param last_ts: iterable of the last time points
        :param last_bs: iterable of the last two state elements
        :param last_ys: iterable of the last ode state variables
        :param simulation_state: object to specify current state of the simulation
        :return:
        """
        pass

    def has_previous_records(self):
        """ Check whether there are records which can be used to continue the simulation
        :return has_previous_records:
        """

        return False

    def load_simulation_state(self):
        """

        :return simulation state: object to reinitialize a running simulation
        """


class StateRecordingMonitor(AbstractMonitor):
    def record(self, t):
        return t > self.ts[-1] + self.interval

    def make_record(self, last_ts, last_bs, last_ys, _=None):
        if self.first_record:
            self.ts = np.array(last_ts)
            self.bs = np.array(last_bs)
            self.ys = np.array(last_ys)
            self.first_record = False
        else:
            self.ts = numpy.concatenate((self.ts, np.array(last_ts)[1:]))
            self.bs = numpy.concatenate((self.bs, np.array(last_bs)[1:]))
            self.ys = numpy.concatenate((self.ys, np.array(last_ys)[1:]))

    def __init__(self, recording_interval):
        self.interval = recording_interval
        self.first_record = True
        self.ts = None
        self.bs = None
        self.ys = None


class GeneralPersistenceMonitor(AbstractMonitor):
    def record(self, t):
        return t > self.last_t + self.interval

    def make_record(self, last_ts, last_bs, last_ys, sim_state=None):
        log.info("Persisting simulation in {} at time {}".format(self.file, last_ts[-1]))
        last_ts = np.array(last_ts, dtype=np.float64)
        last_bs = np.array(last_bs, dtype=np.int32)
        last_ys = np.array(last_ys, dtype=np.float64)
        if self.first_record:
            for data_id, transformation in self.recording_dict.iteritems():
                data = transformation(last_ts, last_bs, last_ys)
                self._create_entry(data_id, data)
            self.first_record = False
        else:
            for data_id, transformation in self.recording_dict.iteritems():
                data = transformation(last_ts[1:], last_bs[1:], last_ys[1:])
                self._extend_entry(data_id, data)
        if sim_state is not None:
            self._save_simulation_state(sim_state)
        self.last_t = last_ts[-1]

    def has_previous_records(self):
        has_previous_records = False
        if os.path.exists(self.file):
            with h5py.File(self.file, "r") as recording_file:
                if "simulation_state" in recording_file.attrs.keys():
                    has_previous_records = True
                    self.first_record = False
        return has_previous_records

    def load_simulation_state(self):
        with h5py.File(self.file, "r") as recording_file:
            simulation_state = dill_pickle.loads(recording_file.attrs["simulation_state"].tostring())
        self.last_t = simulation_state.get_time()
        log.info("Loaded previous simulation state at simulation time {}.".format(self.last_t))
        return simulation_state

    def _save_simulation_state(self, sim_state):
        with h5py.File(self.file, "a") as recording_file:
            dill_pickle.settings['recurse'] = True
            simulation_state_string = dill_pickle.dumps(sim_state, protocol=2)
            recording_file.attrs["simulation_state"] = np.void(simulation_state_string)

    def _create_file(self):
        if not os.path.exists(self.file):
            recording_table = tables.open_file(self.file, 'w')
            recording_table.close()

    def _create_entry(self, data_id, data):
        recording_table = tables.open_file(self.file, 'a')
        recording_table.create_earray(recording_table.root, data_id, obj=data)
        recording_table.close()

    def _extend_entry(self, data_id, data):
        recording_table = tables.open_file(self.file, 'a')
        node = recording_table.get_node(recording_table.root, data_id)
        node.append(data)
        recording_table.close()

    def get_data(self, data_id):
        recording_table = tables.open_file(self.file, 'r')
        node = recording_table.get_node(recording_table.root, data_id)
        data = node[:]
        recording_table.close()
        return data

    def __init__(self, path_to_persistence_file, interval, recording_dictionary):
        self.file = path_to_persistence_file
        self.interval = interval
        self.recording_dict = recording_dictionary
        self.first_record = True
        self.last_t = None
        self._create_file()


class PersistenceMonitor(GeneralPersistenceMonitor):
    def __init__(self, path_to_persisting_file, interval):
        recording_dict = {'t': lambda ts, bs, ys: ts,
                          'b': lambda ts, bs, ys: bs,
                          'y': lambda ts, bs, ys: ys}
        super(PersistenceMonitor, self).__init__(path_to_persisting_file, interval, recording_dict)


def brute(transition_rates, ode_rhs, init_discrete, init_continuous, tmax, ode_dt, out_dt, integrator, integrator_args={}, rng=None):
    """
    Brute force simulation of a special stochastic hybrid system
    The discrete part of the system b in {0,1}^NxM consists of N two state elements organised in subgroups of size M. The transition rates between the two states depend on the state of the other elements in the subgroup and the external continuous variables y.

    :param transition_rates: returns NxM array of transition rates depending on y, b
    :param ode_rhs: returns the derivatives depending on y, b, t
    :param init_discrete: NxM array of the binary starting values of the discrete elements
    :param init_continuous: array with the starting values of the continuous variables
    :param tmax: length of the simulation
    :param ode_dt: stimulation step size
    :param out_dt: time resolution of the returned evolution
    :param integrator: integrator type e.g. 'dopri5'
    :param integrator_args: additional arguments passed to the integrator, check out scipy.integrate.ode for details
    :param rng: Random Number Generator
    :return: time vector, states of the binary systems, evolution of y
    """

    if rng is None:
        rng = np.random.RandomState()

    initial_time = 0

    t = initial_time
    b = init_discrete
    y = init_continuous

    t_out = deque()
    t_out.append(t)
    b_out = deque()
    b_out.append(np.copy(b))
    y_out = deque()
    y_out.append(y)

    ode_solver = ode(ode_rhs).set_integrator(integrator, **integrator_args)
    ode_solver.set_initial_value(y)

    N, M = b.shape

    progress_bar = tqdm(total=tmax, unit='ms')

    while t < tmax:
        while t < t_out[-1] + out_dt:
            # Step forward in the continuous system
            ode_solver.set_f_params(b)
            ode_solver.integrate(t + ode_dt)

            # Step forward in the discrete system
            time_step = ode_solver.t - t
            transition_probabilities = (transition_rates(y, b) * time_step)
            is_transition = (transition_probabilities >= rng.rand(N, M))

            t = ode_solver.t
            update_discrete_system(b, is_transition)
            y = ode_solver.y

        t_out.append(t)
        b_out.append(np.copy(b))
        y_out.append(y)
        progress_bar.update(out_dt)
    progress_bar.close()

    return scipy.array(t_out), scipy.array(b_out, dtype=int), scipy.array(y_out)


def monitored_brute(monitor, transition_rates, ode_rhs, init_discrete, init_continuous, tmax, ode_dt, out_dt, integrator, integrator_args={}, rng=None):
    """
    Brute force simulation of a special stochastic hybrid system
    The discrete part of the system b in {0,1}^NxM consists of N two state elements organised in subgroups of size M. The transition rates between the two states depend on the state of the other elements in the subgroup and the external continuous variables y.

    :param monitor: a state monitor which records configured state-dependent values conditional on time
    :param transition_rates: returns NxM array of transition rates depending on y, b
    :param ode_rhs: returns the derivatives depending on y, b, t
    :param init_discrete: NxM array of the binary starting values of the discrete elements
    :param init_continuous: array with the starting values of the continuous variables
    :param tmax: length of the simulation
    :param ode_dt: stimulation step size
    :param out_dt: time resolution of the returned evolution
    :param integrator: integrator type e.g. 'dopri5'
    :param integrator_args: additional arguments passed to the integrator, check out scipy.integrate.ode for details
    :param rng: Random Number Generator
    :return: time vector, states of the binary systems, evolution of y
    """

    if monitor.has_previous_records():
        simulation_state = monitor.load_simulation_state()
        initial_time = simulation_state.get_time()
        init_discrete = simulation_state.get_discrete_system_state()
        init_continuous = simulation_state.get_continuous_system_state()
        rng = simulation_state.get_random_state_generator()
        ode_solver = simulation_state.get_ode_instance()
        if initial_time > tmax:
            return None
    else:
        if rng is None:
            rng = np.random.RandomState()

        initial_time = 0

    t = initial_time
    b = init_discrete
    y = init_continuous

    t_out = deque()
    t_out.append(t)
    b_out = deque()
    b_out.append(np.copy(b))
    y_out = deque()
    y_out.append(y)
    if not monitor.has_previous_records():
        ode_solver = ode(ode_rhs).set_integrator(integrator, **integrator_args)
        ode_solver.set_initial_value(y)
        current_simulation_state = SimulationState(t, b, y, rng, ode_solver)
        monitor.make_record(t_out, b_out, y_out, current_simulation_state)

    N, M = b.shape
    progress_bar = tqdm(initial=t, total=tmax, unit='ms')
    while t < tmax:
        if monitor.record(t):
            last_t = t_out[-1]
            last_b = b_out[-1]
            last_y = y_out[-1]
            current_simulation_state = SimulationState(last_t, last_b, last_y, rng, ode_solver)
            monitor.make_record(t_out, b_out, y_out, current_simulation_state)
            t_out.clear()
            b_out.clear()
            y_out.clear()
            t_out.append(last_t)
            b_out.append(last_b)
            y_out.append(last_y)

        while t < t_out[-1] + out_dt:
            # Step forward in the continuous system
            ode_solver.set_f_params(b)
            ode_solver.integrate(t + ode_dt)

            # Step forward in the discrete system
            time_step = ode_solver.t - t
            transition_probabilities = (transition_rates(y, b) * time_step)
            is_transition = (transition_probabilities >= rng.rand(N, M))

            t = ode_solver.t
            update_discrete_system(b, is_transition)
            y = ode_solver.y

        t_out.append(t)
        b_out.append(np.copy(b))
        y_out.append(y)
        progress_bar.update(out_dt)

    last_t = t_out[-1]
    last_b = b_out[-1]
    last_y = y_out[-1]
    current_simulation_state = SimulationState(last_t, last_b, last_y, rng, ode_solver)
    monitor.make_record(t_out, b_out, y_out, current_simulation_state)
    progress_bar.close()


def update_discrete_system(b, is_transition):
    b[np.where(is_transition)] = 1 * np.logical_not(b[np.where(is_transition)])
