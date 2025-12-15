from typing import List
from dotenv import load_dotenv
import os
import re

import numpy as np
from google import genai
from google.genai import types

from arc import ArcProblem
from arc.types import ArcIOPair, ArcGrid, ArcPrediction, verify_is_arc_grid
from arc.agents import ArcAgent

import prompts as p
from file import save_results, write_txt

import time


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

    def _init_chat(self, sys_prompt: str) -> None:
        self.chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                temperature = 0,
            )
        )
        self.chat.send_message(sys_prompt)


    def solve(self, task: ArcProblem) -> List[ArcPrediction]:
        sys_prompt = p.SYSTEM + "\n" + p.GENERAL
        self._init_chat(sys_prompt)
        outputs = []
        dir = save_results(self.model)
        tg = task.test_pairs[0]
        predictions = []
        msg = p.build_task(p.SOLVE, task.train_pairs, tg.x)
        write_txt(dir, "in.txt", task.uid + ":\n" + msg)

        # initial attempt
        out = self.chat.send_message(msg)
        grid = parse_grid(out.text)
        predictions.append(grid)
        write_txt(dir, "predict.txt", np.array2string(grid, separator=" "))

        # try again if incorrect
        for _ in range(MAX_ATTEMPTS - 1):
            print("trying again")
            if cmp_grids(grid, tg.y): break
            out = self.chat.send_message(p.RETRY)
            grid = parse_grid(out.text)
            predictions.append(grid)
            write_txt(dir, "predict.txt", np.array2string(grid, separator=" "))
        print("predictions done")
        outputs.append(predictions)
        if cmp_grids(grid, tg.y):
            write_txt(dir, "select.txt", "Skip")
        else: 
            write_txt(dir, "select.txt", self._select(tg.y))
        print("selection done")

    def _select(self, solution):
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
