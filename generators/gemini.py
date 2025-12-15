from file import get_arctask
from agents import Gemini

task = get_arctask("308 c59eb873")

agent = Gemini("gemini-2.5-flash-lite", "KEYC0")
outs = agent.solve(task)
    
