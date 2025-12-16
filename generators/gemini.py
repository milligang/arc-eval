from file import get_arctask, save_rand_arcset
from agents import Gemini

task = get_arctask("arcset1")

agent = Gemini("gemini-2.5-flash-lite", "KEYK1")
outs = agent.correction(task, 10)
    
