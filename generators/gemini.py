from file import get_arcset, save_rand_arcset, get_arctask, get_predictions
from agents import Gemini, cmp_grids
import time

agent = Gemini("gemini-2.5-flash", "KEYK2")
incorrect = get_predictions("solve3", "g25f0")[1][0]
solution = get_arctask("75 6455b5f5")
if not cmp_grids(incorrect, solution.test_pairs[0].y):
    agent.select(incorrect, solution)

def correct(p: int):
    tasks = get_arcset(save_rand_arcset(10))
    for task in tasks:
        agent.correction(task, p)
        time.sleep(60)

def solve():
    tasks = get_arcset(save_rand_arcset(10))
    for task in tasks:
        agent.solve(task)
        time.sleep(60)

    
