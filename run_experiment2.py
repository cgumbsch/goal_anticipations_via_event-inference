import gym
import interaction_gym
import numpy as np
import event_inference as event
import sys
import random


def test_run(directory_name, setting_name, event_system, interaction_env,
             claw, simulation_num, epoch_num, run_num, time_horizon,
             file_name_addition=''):
    """
    Performs one test run and logs the results
    :param directory_name: name of target folder for log files
    :param setting_name: name of this simulation setting
    :param event_system: instance of trained event inference system
    :param interaction_env: instance of agent-patient interaction gym
    :param claw: bool, whether to use claw- or hand-agent in this test run
    :param simulation_num: number of this simulation
    :param epoch_num: number of training phases
    :param run_num: number of runs in this testing phase
    :param time_horizon: tau
    :param file_name_addition: extra string added at the end of file name
    """
    entity_name = 'hand'
    if claw:
        entity_name = 'claw'
    filename = directory_name + '/' + setting_name + str(simulation_num) + '_epoch' + str(
        epoch_num) + "_" + entity_name + file_name_addition + "_run" + str(run_num) + '.txt'
    file = open(filename, 'w')
    file.write('t, Event, Policy, P(still), P(random), P(reach), P(transport), o(t) \n')
    o_t = interaction_env.reset_to_grasping(claw=claw)  # claw=False for hand agent, claw=True for claw agent
    pi_t = np.array([0.0, 0.0, 1.0])  # During testing the system starts with no fixation
    event_system.reset()
    for t in range(270):
        # 1. step: Get o(t)
        o_t, r_t, done_t, info_t = interaction_env.step(pi_t)

        # 2. step: Infer event model and next action
        pi_t, probs = event_system.step(o_t=o_t, pi_t=pi_t, training=False,
                                        done=done_t, e_i=info_t, tau=time_horizon)

        # 3. step: Log data
        obs_str = ', '.join(map(str, o_t))
        file.write(
            str(t) + ', ' + str(info_t) + ', ' + str(np.argmax(pi_t)) + ', ' + str(probs[0]) +
            ', ' + str(probs[1]) + ', ' + str(probs[2]) + ', ' + str(probs[3]) + ', ' + obs_str + '\n')

    file.close()
    interaction_env.close()


# Global parameter settings used for all experiments
epsilon_start = 0.01
epsilon_end = 0.001
epsilon_dynamics = 0.001
random_Colors = True
tau = 2
folder_name = 'Experiments/ResAblationExperience'

# EXPERIMENT 2:
# tau = 2, 50% E_grasp events in training, randomized agent appearance
percentage_reaching = 0.5
test_name = 'res_perc_50_sim'

for simulation in range(10):
    seed = simulation
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)
    for epoch in range(30):
        # TRAINING PHASE:
        # do 100 training event sequences per phase
        for sequence in range(100):
            # reset environment to new event sequence
            observation = env.reset()
            # sample one-hot-encoding of policy pi(0)
            policy_t = np.array([0.0, 0.0, 0.0])
            policy_t[random.randint(0, 2)] = 1.0
            done = False
            while not done:
                # perform pi(t) and receive new observation o(t)
                observation, reward, done, info = env.step(policy_t)
                # update the event probabilities, event schemata, and infer next policy
                policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True, done=done, e_i=info)

        # TESTING PHASE:
        # do 10 test phases for hand and claw agents
        for run in range(10):
            # hand:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=False, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
            # claw:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=True, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)

# tau = 2, 20% E_grasp events in training, randomized agent appearance
percentage_reaching = 0.2
test_name = 'res_perc_20_sim'

for simulation in range(10):
    seed = simulation
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)
    for epoch in range(30):
        # TRAINING PHASE:
        # do 100 training event sequences per phase
        for sequence in range(100):
            # reset environment to new event sequence
            observation = env.reset()
            # sample one-hot-encoding of policy pi(0)
            policy_t = np.array([0.0, 0.0, 0.0])
            policy_t[random.randint(0, 2)] = 1.0
            done = False
            while not done:
                # perform pi(t) and receive new observation o(t)
                observation, reward, done, info = env.step(policy_t)
                # update the event probabilities, event schemata, and infer next policy
                policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True, done=done, e_i=info)

        # TESTING PHASE:
        # do 10 test phases for hand and claw agents
        for run in range(10):
            # hand:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=False, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
            # claw:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=True, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)

# tau = 2, 10% E_grasp events in training, randomized agent appearance
percentage_reaching = 0.1
test_name = 'res_perc_10_sim'

for simulation in range(10):
    seed = simulation
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)
    for epoch in range(30):
        # TRAINING PHASE:
        # do 100 training event sequences per phase
        for sequence in range(100):
            # reset environment to new event sequence
            observation = env.reset()
            # sample one-hot-encoding of policy pi(0)
            policy_t = np.array([0.0, 0.0, 0.0])
            policy_t[random.randint(0, 2)] = 1.0
            done = False
            while not done:
                # perform pi(t) and receive new observation o(t)
                observation, reward, done, info = env.step(policy_t)
                # update the event probabilities, event schemata, and infer next policy
                policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True, done=done, e_i=info)

        # TESTING PHASE:
        # do 10 test phases for hand and claw agents
        for run in range(10):
            # hand:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=False, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
            # claw:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=True, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)

# tau = 2, 1% E_grasp events in training, randomized agent appearance
percentage_reaching = 0.01
test_name = 'res_perc_1_sim'

for simulation in range(10):
    seed = simulation
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)
    for epoch in range(30):
        # TRAINING PHASE:
        # do 100 training event sequences per phase
        for sequence in range(100):
            # reset environment to new event sequence
            observation = env.reset()
            # sample one-hot-encoding of policy pi(0)
            policy_t = np.array([0.0, 0.0, 0.0])
            policy_t[random.randint(0, 2)] = 1.0
            done = False
            while not done:
                # perform pi(t) and receive new observation o(t)
                observation, reward, done, info = env.step(policy_t)
                # update the event probabilities, event schemata, and infer next policy
                policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True, done=done, e_i=info)

        # TESTING PHASE:
        # do 10 test phases for hand and claw agents
        for run in range(10):
            # hand:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=False, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
            # claw:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=True, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)

# tau = 2, 0.1% E_grasp events in training, randomized agent appearance
percentage_reaching = 0.001
test_name = 'res_perc_01_sim'

for simulation in range(10):
    seed = simulation
    model = event.CAPRI(epsilon_start=epsilon_start, epsilon_dynamics=epsilon_dynamics,
                        epsilon_end=epsilon_end, no_transition_prior=0.9, dim_observation=18,
                        num_policies=3, num_models=4, r_seed=seed, sampling_rate=2)
    env = interaction_gym.InteractionEventGym(sensory_noise_base=1.0, sensory_noise_focus=0.01,
                                              r_seed=seed, randomize_colors=random_Colors,
                                              percentage_reaching=percentage_reaching)
    for epoch in range(30):
        # TRAINING PHASE:
        # do 100 training event sequences per phase
        for sequence in range(100):
            # reset environment to new event sequence
            observation = env.reset()
            # sample one-hot-encoding of policy pi(0)
            policy_t = np.array([0.0, 0.0, 0.0])
            policy_t[random.randint(0, 2)] = 1.0
            done = False
            while not done:
                # perform pi(t) and receive new observation o(t)
                observation, reward, done, info = env.step(policy_t)
                # update the event probabilities, event schemata, and infer next policy
                policy_t, P_ei = model.step(o_t=observation, pi_t=policy_t, training=True, done=done, e_i=info)

        # TESTING PHASE:
        # do 10 test phases for hand and claw agents
        for run in range(10):
            # hand:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=False, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
            # claw:
            test_run(directory_name=folder_name, setting_name=test_name, event_system=model,
                     interaction_env=env, claw=True, simulation_num=simulation,
                     epoch_num=epoch, run_num=run, time_horizon=tau)
