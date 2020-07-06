# import submission module to test as agent function
from submissions.submission_dqn import agent_function
from kaggle_environments import evaluate, make, utils

def mean_reward(rewards):
    return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))


reward_random = mean_reward(evaluate("connectx", [agent_function, "random"], num_episodes=100))
reward_negamax =  mean_reward(evaluate("connectx", [agent_function, "negamax"], num_episodes=100))
reward_random_inv = mean_reward(evaluate("connectx", ["random", agent_function], num_episodes=100))
reward_negamax_inv =  mean_reward(evaluate("connectx", ["negamax", agent_function], num_episodes=100))

print("Ours vs Negamax:", reward_negamax)
print("Ours vs Random:",reward_random)
print("Negamax vs Ours:", reward_negamax_inv)
print("Random vs Ours:",reward_random_inv)
