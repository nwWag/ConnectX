import inspect
import os

no_params_path = "submission_template.py"
def append_object_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)
        
def write_agent_to_file(function, file):
    with open(file, "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(agent_function, no_params_path)

# Write net parameters to submission file

import base64
import sys


INPUT_PATH = "Q_Network.py"
OUTPUT_PATH = 'submission.py'

with open(INPUT_PATH, 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes).decode()

with open(no_params_path, 'r') as file:
    data = file.read()

data = data.replace('BASE64_PARAMS', encoded_weights)

with open(OUTPUT_PATH, 'w') as f:
    f.write(data)
    print('written agent with net parameters to', OUTPUT_PATH)