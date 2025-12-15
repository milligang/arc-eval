from arc.types import ArcIOPair
from arc import train_problems, ArcProblem
from agents import RandomAgent, Gemini

import random


# get a random problem
train_problems : list[ArcProblem]
task : ArcProblem = random.choice(train_problems)
print(task)
# display chosen problem
for i, pair in enumerate(task.train_pairs, start=1):
    pair.plot(show=True, title=f"Demo {i}")

for i, pair in enumerate(task.test_pairs, start=1):
    pair.plot(show=True, title=f"Test {i}")

# run on random agent
agent = RandomAgent()
outs = agent.predict(task.train_pairs, task.test_inputs)

for test_pair, predicitons in zip(task.test_pairs, outs):
    for p in predicitons:
        prediction = ArcIOPair(test_pair.x, p)
        prediction.plot(show=True)

# run on gemini agent
# agent = Gemini("gemini-2.5-flash-lite")
# outs = agent.solve(task)
# print("Final output:\n", outs)
# for test_pair, predicitons in zip(task.test_pairs, outs):
#     for p in predicitons:
#         prediction = ArcIOPair(test_pair.x, p)
#         prediction.plot(show=True)