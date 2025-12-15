from file import save_rand_arcset, get_arcset
from agents import Gemini
import time

tasks = get_arcset(save_rand_arcset(3))

agent = Gemini("gemini-2.5-flash-lite")
for task in tasks:
    outs = agent.solve(task)
    time.sleep(60)
    
