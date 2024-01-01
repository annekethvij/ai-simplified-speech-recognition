from collections import defaultdict

def strip_quotes(s):
    return s.strip('"')

def read_observation_actions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_pairs = int(lines[1])

    observations_actions = []
    for line in lines[2:]:
        parts = line.strip().split()
        observation = strip_quotes(parts[0])
        action = strip_quotes(parts[1]) if len(parts) > 1 else None
        observations_actions.append((observation, action))

    # print("num_pairs: ", num_pairs)
    # print("observations_actions: ", observations_actions)

    return num_pairs, observations_actions

def read_transition_weights(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    _, _, _, default_weight = map(int, lines[1].split())
    transition_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_weight)))

    for line in lines[2:]:
        state1, action, state2, weight = line.strip().split()
        transition_weights[strip_quotes(state1)][strip_quotes(action)][strip_quotes(state2)] = int(weight)

    for state in transition_weights:
        for action in transition_weights[state]:
            total_weight = sum(transition_weights[state][action].values())
            for state2 in transition_weights[state][action]:
                transition_weights[state][action][state2] /= total_weight

    # print("transition_weights: ", transition_weights)
    return transition_weights

def read_initial_probs(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_states = int(lines[1].split()[0])
    weights = {strip_quotes(line.split()[0]): float(line.split()[1]) for line in lines[2:]}

    total_weight = sum(weights.values())
    initial_probs = {state: weight / total_weight for state, weight in weights.items()}

    return num_states, initial_probs

def read_state_observation_weights(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    _, _, _, default_weight = map(int, lines[1].split())
    weights = defaultdict(lambda: defaultdict(lambda: default_weight))

    for line in lines[2:]:
        state, observation, weight = line.strip().split()
        weights[strip_quotes(state)][strip_quotes(observation)] = int(weight)

    probabilities = defaultdict(lambda: defaultdict(float))
    for state in weights:
        total_weight = sum(weights[state].values())
        for observation in weights[state]:
            probabilities[state][observation] = weights[state][observation] / total_weight

    return probabilities


def viterbi_algorithm(observations, initial_probabilities, obs_probabilities, trans_weights):
    num_observations = len(observations)
    states = list(initial_probabilities.keys())
    probabilities_matrix = [{}]
    state_path = {}

    for state in states:
        obs_prob = obs_probabilities[state].get(observations[0][0], 0)
        probabilities_matrix[0][state] = initial_probabilities[state] * obs_prob
        state_path[state] = [state]
        # print(f"Initial probability for state {state}: {probabilities_matrix[0][state]}")

    for obs_index in range(1, num_observations):
        probabilities_matrix.append({})
        new_path = {}

        for current_state in states:
            action = observations[obs_index-1][1] if observations[obs_index-1][1] else None
            max_prob, previous_state = None, None
            for prev_state in states:
                transition_prob = trans_weights[prev_state].get(action, {}).get(current_state, 0)
                prob = probabilities_matrix[obs_index-1][prev_state] * transition_prob * obs_probabilities[current_state].get(observations[obs_index][0], 0)
                # print(f"Transition from {prev_state} to {current_state} with action '{action}' has probability {prob}")
                if max_prob is None or prob > max_prob:
                    max_prob, previous_state = prob, prev_state

            probabilities_matrix[obs_index][current_state] = max_prob
            new_path[current_state] = state_path[previous_state] + [current_state]
            # print(f"Max probability for state {current_state}: {max_prob}")
        state_path = new_path
    max_prob, final_state = None, None

    for state in states:
        prob = probabilities_matrix[num_observations-1][state]
        if max_prob is None or prob > max_prob:
            max_prob, final_state = prob, state

    # print(f"Final most probable state: {final_state} with probability {max_prob}")
    return state_path[final_state]

def write_states_to_file(states, file_path):
    with open(file_path, 'w') as file:
        file.write("states\n")
        file.write(f"{len(states)}\n")
        for state in states:
            file.write(f'"{state}"\n')

if __name__ == "__main__":
    observation_actions_file = 'observation_actions.txt'
    transition_weights_file = 'state_action_state_weights.txt'
    initial_probs_file = 'state_weights.txt'
    observation_weights_file = 'state_observation_weights.txt'

    num_pairs, observations_actions = read_observation_actions(observation_actions_file)
    transition_weights = read_transition_weights(transition_weights_file)
    num_states, initial_probs = read_initial_probs(initial_probs_file)
    probabilities = read_state_observation_weights(observation_weights_file)

    most_likely_states = viterbi_algorithm(observations_actions, initial_probs, probabilities, transition_weights)
    # print("\nMost Likely Sequence: \n", most_likely_states)

    write_states_to_file(most_likely_states, 'states.txt')