# run on gemini agent
agent = Gemini("gemini-2.5-flash-lite")
outs = agent.solve(task)
print("Final output:\n", outs)
for test_pair, predicitons in zip(task.test_pairs, outs):
    for p in predicitons:
        prediction = ArcIOPair(test_pair.x, p)
        prediction.plot(show=True)