from kaggle_environments import evaluate, make, utils
# Change submission to get another agent
from submissions.submission_dqn import agent_function

# Setup enviroment
env = make("connectx", debug=True)
# Create game and opponent
trainer = env.train([None, agent_function])

# Play a game
observation = trainer.reset()
while not env.done:
    # Show game board and get user input
    env.render()
    action = int(input("Column ? "))

    # Verify user input
    if action not in [1,2,3,4,5,6,7]:
        action = 0
    else:
        action -= 1

    # Play a step
    observation, reward, done, info = trainer.step(action)

    # Check if game has finished
    if done:
        print("You WON. SUPI DUPI." if env.state[0]['reward'] == 1 else "Nevermind. Next time.")



