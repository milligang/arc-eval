from file import get_arcset, save_rand_arcset
from agents import Gemini
import time

tasks = get_arcset(save_rand_arcset(20))

agent = Gemini("gemini-2.5-flash-lite", "KEYK2")
for task in tasks:
    outs = agent.correction(task, 10)
    time.sleep(60)
    
