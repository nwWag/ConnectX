from kaggle_environments import evaluate, make, utils


env = make("connectx", debug=True)
env.render()

def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

env.reset()
# Play as the first agent against default "random" agent.
env.run([my_agent, "random"])
env.render(mode="human", width=500, height=450)


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    print(observation, env.configuration)
    my_action = my_agent(observation, env.configuration)
    print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()

def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))


# "None" represents which agent you'll manually play as (first or second player).
env.play([None, "negamax"], width=500, height=450)