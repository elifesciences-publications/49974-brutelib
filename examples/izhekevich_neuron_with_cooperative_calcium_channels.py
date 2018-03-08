import matplotlib.pyplot as plt
import numpy as np

from brutelib import brute


def get_stimulation_current():
    t_on = 10
    t_off = 50
    amplitude = 15
    baseline = 0.0

    def stimulation_current(t):
        current = baseline
        if t_on < t <= t_off:
            current = amplitude
        return current

    return stimulation_current


def get_neuron_ode(input_current):
    # Neuron parameter
    C_m = 1
    g_K = 10
    E_K = -90
    g_Na = 20
    E_Na = 60
    g_L = 8
    E_L = -80

    # Cooperative channels
    g = 0.06
    E = 100

    def rhs(t, y, b):
        v = y[0]
        n = y[1]

        ratio_of_open_channels = np.sum(b) / np.prod(np.shape(b))
        cluster_current = ratio_of_open_channels * g * (v - E)

        m = 1. / (1 + np.exp((-20 - v) / 15.))
        dv = 1.0 / C_m * (
            input_current(t) - g_K * n * (v - E_K) - g_Na * m * (v - E_Na) - g_L * (v - E_L) - cluster_current)
        dn = (1. / (1 + np.exp((-25 - v) / 5.)) - n) / 0.5
        return np.array([dv, dn])

    return rhs


def get_channel_transition_rates(cluster_size, cluster_number, cluster_interaction, opening_rate, closing_rate):
    channel_interaction = cluster_interaction / float(cluster_size - 1)
    cooperative_opening_rate = lambda v, open_channels: opening_rate(v + open_channels * channel_interaction)
    cooperative_closing_rate = lambda v, open_channels: closing_rate(v + (open_channels - 1) * channel_interaction)

    def transitions_rates(y, b):
        v = y[0]
        open_channels = np.sum(b, axis=1)
        opening_rates = np.repeat(cooperative_opening_rate(v, open_channels)[:, np.newaxis], b.shape[1], axis=1)
        closing_rates = np.repeat(cooperative_closing_rate(v, open_channels - 1)[:, np.newaxis], b.shape[1], axis=1)
        return np.where(b, closing_rates, opening_rates)

    return transitions_rates


def get_single_channel_rates():
    v_half = -45
    k = 10.0
    tau = 30.0

    def m(v):
        return (1 + np.tanh((v - v_half) / k)) / 2.0

    opening_rate = lambda v: m(v) / tau
    closing_rate = lambda v: (1 - m(v)) / tau
    return opening_rate, closing_rate


def run():
    # Stimulation
    stimulation_current = get_stimulation_current()
    print stimulation_current(0)
    print stimulation_current(150)

    # Channel properties
    opening_rate, closing_rate = get_single_channel_rates()
    print opening_rate(-30)
    print closing_rate(-30)

    # Cluster properties
    cluster_size = 4
    cluster_number = 10
    cluster_interaction = 70

    cluster_initial_state = np.zeros((cluster_number, cluster_size))

    channel_transition_rates = get_channel_transition_rates(cluster_size,
                                                            cluster_number, cluster_interaction,
                                                            opening_rate, closing_rate)

    # Neuron ode
    v_0 = -65
    n_0 = 0
    neuron_initial_state = np.array([v_0, n_0])
    ode_rhs = get_neuron_ode(stimulation_current)
    print ode_rhs(3, neuron_initial_state, cluster_initial_state)

    # Simulation
    t_max = 150
    ode_dt = 0.005
    out_dt = 0.1
    integrator = 'dopri5'
    integrator_args = {"method": "bdf"}
    rng = np.random.RandomState(2)

    # Run simulation
    t, channel_states, neuron_states = brute(channel_transition_rates, ode_rhs, cluster_initial_state,
                                             neuron_initial_state, t_max, ode_dt, out_dt, integrator, integrator_args,
                                             rng)

    # Plot
    fig = plt.figure()
    ax_stimulus = fig.add_subplot(411)
    ax_stimulus.plot(t, np.array([stimulation_current(time) for time in t]))

    ax_v = fig.add_subplot(412)
    ax_v.plot(t, neuron_states[:, 0])

    ax_conductance = fig.add_subplot(413)
    ax_conductance.plot(t, np.sum(np.sum(channel_states, axis=2), axis=1))

    ax_channels = fig.add_subplot(414)
    reshaped_channel_states = np.transpose(
        np.reshape(channel_states, (channel_states.shape[0], np.prod(channel_states.shape[1:]))))

    ax_channels.imshow(reshaped_channel_states, aspect='auto', extent=(t[0], t[-1], 0, cluster_size*cluster_number))
    ax_channels.hlines(np.arange(0,cluster_size*cluster_number,cluster_size), 0, t[-1], linestyles='dashed', color='k', lw = 2)

    ax_stimulus.set_ylabel('Injected current[uA/cm**2]')
    ax_v.set_ylabel('Voltage [mV]')
    ax_conductance.set_ylabel('# open channels')
    ax_channels.set_ylabel('Channel states')
    ax_channels.set_xlabel('t[ms]')
    plt.show()


run()
