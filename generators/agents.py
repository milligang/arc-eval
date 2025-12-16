from typing import List
from dotenv import load_dotenv
import os
import re
import random

import numpy as np
from google import genai
from google.genai import types

from arc import ArcProblem
from arc.types import ArcIOPair, ArcGrid, ArcPrediction, verify_is_arc_grid
from arc.agents import ArcAgent

import prompts as p
from file import save_results, write_txt


load_dotenv()
MAX_ATTEMPTS = 3


def parse_grid(response) -> np.ndarray:
    """
    Parse an ARC grid from an LLM response.
    Accepts common formats:
      - Python-style lists
      - Space-separated rows
    """
    text = response.strip()
    # Try Python list syntax
    try:
        grid = eval(text, {"__builtins__": {}})
        arr = np.array(grid, dtype=int)
        verify_is_arc_grid(arr)
        return arr
    except Exception:
        pass

    # Try whitespace / line-based grid
    rows = []
    for line in text.splitlines():
        nums = re.findall(r"\d", line)
        if nums:
            rows.append([int(n) for n in nums])

    if rows:
        arr = np.array(rows, dtype=int)
        verify_is_arc_grid(arr)
        return arr

    raise ValueError(
        "Failed to parse ARC grid from model output.\n"
        f"Raw output:\n{text}"
    )


def cmp_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> bool:
    """
    ret true if same shape and identical values.
    """
    verify_is_arc_grid(grid_a)
    verify_is_arc_grid(grid_b)
    return grid_a.shape == grid_b.shape and np.all(grid_a == grid_b)


def corrupt_grid(grid: np.ArcGrid, percent: int) -> ArcGrid:
    h, w = grid.shape
    total = h * w
    p = min(100, min(0, percent))
    num_corrupt = int((p / 100) * total)

    corrupted = grid
    random_indices = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(num_corrupt)]
    for (r, c) in random_indices:
        corrupted[r][c] = random.randint(1, 9)

    verify_is_arc_grid(corrupted)
    return corrupted

class RandomAgent(ArcAgent):
    """Makes random predicitons. Low chance of success. """

    def predict(
            self, demo_pairs: List[ArcIOPair], test_grids: List[ArcGrid]
    ) -> List[ArcPrediction]:
        """We are allowed to make up to 3 guesses per challange rules. """
        outputs = []
        for tg in test_grids:
            a = p.build_task(p.SOLVE, demo_pairs, tg)
            out_shape = tg.shape
            out1 = np.random.randint(0, 9, out_shape)
            out2 = np.random.randint(0, 9, out_shape)
            out3 = np.random.randint(0, 9, out_shape)
            outputs.append([out1, out2, out3])
        return outputs


class Gemini(ArcAgent):
    def __init__(self, model: str, key_name: str):
        api_key = os.getenv(key_name)
        if api_key is None:
            raise ValueError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.chat = None
        self.model = model
        self.dirc = ""

    def _init_chat(self) -> None:
        self.chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                temperature = 0,
            )
        )

    def predict(self, msg: str, task: ArcProblem):
        predictions = []
        # initial attempt
        out = self.chat.send_message(msg)
        grid = parse_grid(out.text)
        predictions.append(grid)
        write_txt(self.dirc, "predict.txt", np.array2string(grid, separator=" "))

        # try again if incorrect
        for _ in range(MAX_ATTEMPTS - 1):
            print("trying again")
            if cmp_grids(grid, task.y): break
            out = self.chat.send_message(p.RETRY)
            grid = parse_grid(out.text)
            predictions.append(grid)
            write_txt(self.dirc, "predict.txt", np.array2string(grid, separator=" "))
        print("predictions done")
        return grid
    
    def solve(self, task: ArcProblem):
        # start new chat
        self._init_chat()
        sys_prompt = p.SYSTEM + "\n" + p.GENERAL
        self.chat.send_message(sys_prompt)

        self.dirc = save_results(self.model, "solve")
        # only handle first test case (very few have more than 1 anyways)
        tg = task.test_pairs[0]
        msg = p.build_task(p.SOLVE, task.train_pairs, tg.x)
        write_txt(self.dirc, "in.txt", task.uid + ":\n" + msg)

        grid = self.predict(msg, tg)
        
        # If incorrect, see if model can recognize a correct solution
        if cmp_grids(grid, tg.y):
            write_txt(self.dirc, "select.txt", "Skip")
        else: 
            write_txt(self.dirc, "select.txt", self._select_solved(tg.y))
        print("selection done")

    def correction(self, task: ArcProblem, percent: int):
        self._init_chat()
        self.dirc = save_results(self.model, "crtxn")
        tg = task.test_pairs[0]
        corrupted = corrupt_grid(tg.y, percent)
        
        sys_prompt = p.SYSTEM + p.GENERAL
        msg = p.build_task(p.CORRECTION, task.train_pairs, tg.x)
        incorrect = "\n".join(" ".join(map(str, row)) for row in corrupted)
        msg = "\n".join([sys_prompt, msg, incorrect])
        write_txt(self.dirc, "in.txt", (task.uid + msg))

        self.predict(msg, task.test_pairs[0])

    def select(self, incorrect, task: ArcProblem):
        self._init_chat()
        self.dirc = save_results(self.model, "choose")
        
        test = task.test_pairs[0]
        select_msg = p.SELECT_TWO(incorrect, test.y)
        msg = p.SYSTEM + p.GENERAL + p.build_task(select_msg, task.train_pairs, test.x)
        write_txt(self.dirc, "in.txt", (task.uid + msg))

        a_or_b = self.chat.send_message(msg)
        write_txt(self.dirc, "predict.txt", a_or_b.text)

    def _select_solved(self, solution):
        a_or_b = self.chat.send_message(p.SELECT(solution))
        return a_or_b.text

    def _categorize_solved(self):
        cat = self.chat.send_message(p.CATEGORIZE_SOLVED)
        return cat.text

    def categorize(self, demo_pairs: List[ArcIOPair]):
        return
    
class Mistral(ArcAgent):
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
