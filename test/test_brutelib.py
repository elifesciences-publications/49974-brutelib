import os
from unittest import TestCase

import numpy as np
from brutelib.brutelib import update_discrete_system, brute, monitored_brute, \
    StateRecordingMonitor, PersistenceMonitor


class TestBrute(TestCase):
    fixed_rate = 0.1
    number_of_continuous_variables = 2

    @staticmethod
    def transition_rates(y, b):
        return TestBrute.fixed_rate

    @staticmethod
    def ode_rhs(t, y, b):
        return np.zeros(TestBrute.number_of_continuous_variables)

    def setUp(self):
        self.n = 1
        self.m = 2
        self.init_discrete = np.zeros((self.m, self.n))
        self.init_continuous = np.zeros(TestBrute.number_of_continuous_variables)
        self.transition_rates = TestBrute.transition_rates
        self.ode_rhs = TestBrute.ode_rhs
        self.tmax = 1
        self.ode_dt = 0.001
        self.out_dt = 0.1
        self.integrator = 'dopri5'
        self.integrator_args = {'method': 'bdf'}

    def test_update_discrete_system(self):
        b = np.array([[0, 1], [1, 1]])
        is_transition = np.array([[True, False], [True, False]])
        expected_b = np.array([[1, 1], [0, 1]])
        update_discrete_system(b, is_transition)
        np.testing.assert_array_equal(expected_b, b)

    def test_brute_returns_correct_time_vector(self):
        t, b, y = brute(self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous, self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args)
        expected_t = np.arange(0, self.tmax + self.out_dt, self.out_dt)
        np.testing.assert_array_almost_equal(expected_t, t)

    def test_brute_returns_correctly_formatted_continuous_variable_vector(self):
        t, b, y = brute(self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous, self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args)
        expected_y_shape = (len(t), TestBrute.number_of_continuous_variables)
        self.assertEqual(expected_y_shape, y.shape)

    def test_brute_returns_correctly_formatted_discrete_variable_vector(self):
        t, b, y = brute(self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous, self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args)
        expected_b_shape = (len(t), self.m, self.n)
        self.assertEqual(expected_b_shape, b.shape)

    def test_monitored_brute_with_simple_state_recording_monitor_produces_the_same_result(self):
        monitor = StateRecordingMonitor(self.tmax / 2.0)
        rng = np.random.RandomState(1)
        t, b, y = brute(self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous, self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args, rng=rng)
        rng = np.random.RandomState(1)
        monitored_brute(monitor, self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous,
                        self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args, rng=rng)
        np.testing.assert_array_equal(t, monitor.ts)
        np.testing.assert_array_equal(b, monitor.bs)
        np.testing.assert_array_equal(y, monitor.ys)

    def test_monitored_brute_with_persisting_state_monitor_records_correctly(self):
        path_to_persisting_file = 'test_brute_persistence.hdf5'
        monitor = PersistenceMonitor(path_to_persisting_file, self.tmax / 2.0)
        rng = np.random.RandomState(1)
        t, b, y = brute(self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous, self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args, rng=rng)
        rng = np.random.RandomState(1)
        monitored_brute(monitor, self.transition_rates, self.ode_rhs, self.init_discrete, self.init_continuous,
                        self.tmax,
                        self.ode_dt, self.out_dt, self.integrator, self.integrator_args, rng=rng)
        actual_t = monitor.get_data('t')
        actual_b = monitor.get_data('b')
        actual_y = monitor.get_data('y')
        if not os.path.exists(path_to_persisting_file):
            self.fail('No file created by persistence monitor.')
        else:
            os.remove(path_to_persisting_file)

        np.testing.assert_array_equal(t, actual_t)
        np.testing.assert_array_equal(b, actual_b)
        np.testing.assert_array_equal(y, actual_y)
