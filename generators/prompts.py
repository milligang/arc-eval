from typing import List
from arc.types import ArcIOPair, ArcGrid

# System
SYSTEM = """
You are an ARC solver.
Do NOT explain your reasoning unless explicitly asked.
You will be given a list of input-output pairs, labeled “Case 0”, “Case 1”, and so on.
Each grid is a 2D grid of integers 0-9 representing a visual grid. 
There is a SINGLE rule that transforms each input grid to the corresponding output grid.
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
Generate the Output grid that corresponds to the last given Input grid, using the transformation rule you induced from the previous input-output pairs. Output ONLY the resulting grid as a JSON array of arrays. Do not include explanations or extra text.
"""
RETRY = """
Your answer does not solve the puzzle. Try again.
"""
CATEGORIZE = """
Select the SINGLE category that best fits the transformation rule. Output ONLY the category name from the list.
"""

CATEGORIZE_SOLVED = """
Select the SINGLE category that best fits the transformation rule. 
The category must be from this list: [”AboveBelow”, “Center”, “CleanUp”, “CompleteShape”, “Copy”, “Count”, “ExtendToBoundary”, “ExtractObjects”, “FilledNotFilled”, “HorizontalVertical”, “InsideOutside”, “MoveToBoundary”, “Order”, “SameDifferent”, “TopBottom2D”, “TopBottom3D”].
"""

def SELECT(solution):
    txt_solution = "\n".join(" ".join(map(str, row)) for row in solution)
    return f"""
        Your final proposed output grid is labeled "A". The proposed the output grid labeled “B” is {txt_solution}. Which grid is correct? Output ONLY one character: "A" or "B".
        """

def build_task(task_prompt: str, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid] = None) -> str:
    return task_prompt