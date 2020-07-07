from kaggle_environments import evaluate, make, utils
from submissions.submission_dqn import agent_function
env = make("connectx", debug=True)
trainer = env.train([None, "random"])
observation = trainer.reset()
while not env.done:
    env.render()
    action = int(input("Column ? "))

    if action not in [1,2,3,4,5,6,7]:
        action = 0
    else:
        action -= 1

    observation, reward, done, info = trainer.step(action)

    if done:
        print("You WON. SUPI DUPI." if env.state[0]['reward'] == 1 else "Nevermind. Next time.")



