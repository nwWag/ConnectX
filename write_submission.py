import inspect
import os
from REINFORCE_agent_function import agent_function
import agents
import base64
import sys

# Adapted from https://www.kaggle.com/matant/pytorch-dqn-connectx#Write-Submission-File

agent_function_path = "REINFORCE_agent_function.py"   # Construct one function that contains all our agent needs
params_path = "model/REINFORCE_params.pth"              # Give path of network parameters
outfile_path = 'submissions/submission_REINFORCE.py'  # Where to write submission file

with open(params_path, 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes).decode()

with open(agent_function_path, 'r') as file:
    data = file.read()

data = data.replace('BASE64_PARAMS', encoded_weights)
with open(outfile_path, 'w') as f:
    f.write(data)
    print('Written!')
