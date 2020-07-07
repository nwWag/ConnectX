from kaggle_environments import evaluate, make, utils
from submissions.submission_dqn import agent_function
env = make("connectx", debug=True)
env.play([None, agent_function], width=500, height=450)