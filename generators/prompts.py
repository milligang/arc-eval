from typing import List
from arc.types import ArcIOPair, ArcGrid
import numpy as np

# System
SYSTEM = """
You are an ARC solver.
NEVER explain your reasoning.
You will be given a list of Input/Output grid pairs, and pairs are seperated blank lines.
Each grid is a 2D array of integers 0-9 representing a visual grid. 
There is a SINGLE rule that transforms each Input grid to the corresponding Output grid.
"""

# Context
GENERAL = """
The rule may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.
"""
CONCEPT_ARC = """
The rule falls into exactly ONE of these categories: 
[”AboveBelow”, “Center”, “CleanUp”, “CompleteShape”, “Copy”, “Count”, “ExtendToBoundary”, “ExtractObjects”, “FilledNotFilled”, “HorizontalVertical”, “InsideOutside”, “MoveToBoundary”, “Order”, “SameDifferent”, “TopBottom2D”, “TopBottom3D”]
"""


# Instruction
SOLVE = """ 
Generate the Output grid that corresponds to the last given Input grid, using the transformation rule you induced from the previous input-output pairs. 
Output ONLY the resulting grid as a 2D Array. NO explanation.
"""
RETRY = """
Incorrect, try again. Give NO explanation, only your final answer.
"""
CATEGORIZE = """
Select the SINGLE category from the list that best fits the transformation rule. Give NO explanation, only your final answer.
"""

CATEGORIZE_SOLVED = """
Select the SINGLE category that best fits the transformation rule. 
The category must be from this list: [”AboveBelow”, “Center”, “CleanUp”, “CompleteShape”, “Copy”, “Count”, “ExtendToBoundary”, “ExtractObjects”, “FilledNotFilled”, “HorizontalVertical”, “InsideOutside”, “MoveToBoundary”, “Order”, “SameDifferent”, “TopBottom2D”, “TopBottom3D”].
Give NO explanation, only your final answer.
"""

def SELECT(solution):
    txt_solution = "\n".join(" ".join(map(str, row)) for row in solution)
    return f"""
        Your final proposed output grid is labeled "A". The proposed the output grid labeled “B” is {txt_solution}. Which grid is correct, A or B? Why?
        """

def build_task(task_prompt: str, demo_pairs: List[ArcIOPair], test_grid: ArcGrid = None) -> str:
    lines = [] 
    for pair in demo_pairs:
        lines.append("Input:")
        lines.append(np.array2string(pair.x, separator=" "))
        lines.append("Output:")
        lines.append(np.array2string(pair.y, separator=" "))
        lines.append("")  # blank line separates examples

    if test_grid is not None:
        lines.append("Input:")
        lines.append(np.array2string(test_grid, separator=" "))
        lines.append("Output: ?")

    lines.append(task_prompt)
    prompt = "\n".join(lines) 
    return prompt