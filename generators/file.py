import os
import random
from arc import train_problems

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root
BASE_RESULTS = os.path.join(BASE_DIR, "data", "results")
BASE_TASKS = os.path.join(BASE_DIR, "data", "tasks")

def next_dir(base_dir: str, prefix: str):
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and d[len(prefix):].isdigit()
    ]
    nums = [int(d[len(prefix):]) for d in existing]
    k = max(nums) + 1 if nums else 0

    name = f"{prefix}{k}"
    return os.path.join(base_dir, name), name

def save_results(model: str):
    dir = BASE_RESULTS
    if model == "gemini-2.5-flash":
        dir += "g25f0"
    elif model == "gemini-2.5-flash-lite":
        dir += "g25f1"
    else:
        return None
    run_dir, _ = next_dir(dir, "solve")
    os.makedirs(run_dir)
    return run_dir

def write_txt(run_dir: str, fname: str, content: str):
    with open(os.path.join(run_dir, fname), "a") as f:
        f.write(content + "\n\n")

def save_rand_arcset(n: int):
    dir, name = next_dir(BASE_TASKS, "arcset")
    if n > len(train_problems):
        raise ValueError(f"Requested {n} tasks, but only {len(train_problems)} available.")

    sampled_tasks = random.sample(list(enumerate(train_problems)), n)
    with open(dir, "w") as f:
        for idx, task in sampled_tasks:
            f.write(f"{idx} {task.uid}\n")
    return name

def get_arcset(file_name: str):
    path = os.path.join(BASE_TASKS, file_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No arcset file found at: {path}")

    tasks = []
    with open(path, "r") as f:
        for line in f:
            tasks.append(get_arctask(line))

    return tasks

def get_arctask(header: str):
    idx_str, uid = header.strip().split()
    idx = int(idx_str)
    task = train_problems[idx]
    if task.uid != uid:
        raise ValueError(f"UID mismatch for index {idx}: expected {uid}, got {task.uid}")
    return task