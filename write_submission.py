import inspect
import os
from dqn_agent_function import agent_function
import agents
import base64
import sys

# Adapted from https://www.kaggle.com/matant/pytorch-dqn-connectx#Write-Submission-File

agent_function_path = "dqn_agent_function.py"   # Cosntruct one function that contains all our agend needs
params_path = "model/Q_Network.pt"              # Give path of entwork paramters
outfile_path = 'submissions/submission_dqn.py'  # Where to write submission file

with open(params_path, 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes).decode()

with open(agent_function_path, 'r') as file:
    data = file.read()

data = data.replace('BASE64_PARAMS', encoded_weights)
with open(outfile_path, 'w') as f:
    f.write(data)
    print('Written!')