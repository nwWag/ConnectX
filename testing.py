# import submission module to test as agent function
from submissions.submission_dqn import agent_function
from kaggle_environments import evaluate, make, utils

def mean_reward(rewards, first=True):
    return sum(1 if r[0 if first else 1] == 1 else 0 for r in rewards) / float(len(rewards))


reward_random = mean_reward(evaluate("connectx", [agent_function, "random"], num_episodes=10))
print("Ours vs Random:", reward_random)
reward_negamax =  mean_reward(evaluate("connectx", [agent_function, "negamax"], num_episodes=10))
print("Ours vs Negamax:", reward_negamax)
reward_random_inv = mean_reward(evaluate("connectx", ["random", agent_function], num_episodes=10), first=False)
print("Random vs Ours:", reward_random_inv)
reward_negamax_inv =  mean_reward(evaluate("connectx", ["negamax", agent_function], num_episodes=10), first=False)
print("Negamax vs Ours:", reward_negamax_inv)
