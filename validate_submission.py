import sys
from kaggle_environments import evaluate, make, utils
#from  https://www.kaggle.com/matant/pytorch-dqn-connectx#Write-Submission-File
out = sys.stdout
infile_path = "submissions/submission_REINFORCE.py"
try:
    submission = utils.read_file(infile_path)
    agent = utils.get_last_callable(submission)
finally:
    sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")