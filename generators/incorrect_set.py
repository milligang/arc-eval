import argparse
import json
import os
import random
import shutil

# Example: python3 incorrect_set.py --n 30 --p 10

def next_run_dir(base_dir, prefix="rand"):
    """
    Returns (run_name, run_path) for the next available randK directory.
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and d[len(prefix):].isdigit()
    ]
    nums = [int(d[len(prefix):]) for d in existing]
    k = max(nums) + 1 if nums else 0

    run_name = f"{prefix}{k}"
    return run_name, os.path.join(base_dir, run_name)


def gen_random_set(n=20, base_path="../data/tasks/arc_agi_2"):
    """
    Populate a new rand_sets/randK/ with N random tasks from evaluation/.
    Returns the run name (e.g., 'rand0').
    """
    evaluation_path = os.path.join(base_path, "evaluation")
    random_sets_root = os.path.join(base_path, "rand_sets")

    if not os.path.isdir(evaluation_path):
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_path}")

    run_name, run_dir = next_run_dir(random_sets_root)
    os.makedirs(run_dir)

    all_tasks = [
        f for f in os.listdir(evaluation_path)
        if f.endswith(".json")
    ]

    if len(all_tasks) < n:
        raise ValueError(f"Requested {n} tasks but only {len(all_tasks)} available.")

    sampled = random.sample(all_tasks, n)

    for task in sampled:
        src = os.path.join(evaluation_path, task)
        dst = os.path.join(run_dir, task)
        shutil.copyfile(src, dst)

    return run_name


def gen_incorrect(run_name, percentage=10, base_path="../data/tasks"):
    """
    Reads tasks from arc_agi_2/rand_sets/run_name/
    Writes corrupted outputs to incorrect/run_name/
    """
    random_set_path = os.path.join(
        base_path, "arc_agi_2", "rand_sets", run_name
    )
    out_dir = os.path.join(base_path, "incorrect", run_name)

    if not os.path.isdir(random_set_path):
        raise FileNotFoundError(f"Random set not found: {random_set_path}")

    os.makedirs(out_dir, exist_ok=False)

    for fname in os.listdir(random_set_path):
        if not fname.endswith(".json"):
            continue

        in_path = os.path.join(random_set_path, fname)
        out_path = os.path.join(out_dir, fname)

        with open(in_path, "r") as f:
            data = json.load(f)

        for test_case in data.get("test", []):
            output = test_case["output"]
            h, w = len(output), len(output[0])
            total = h * w
            k = int((percentage / 100.0) * total)

            for idx in random.sample(range(total), k):
                r, c = divmod(idx, w)
                output[r][c] = random.randint(1, 9)

        with open(out_path, "w") as f:
            json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random ARC task sets and incorrect solutions"
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Number of tasks to sample")
    parser.add_argument("--p", type=int, required=True,
                        help="Percentage of output cells to corrupt")

    args = parser.parse_args()

    run_name = gen_random_set(args.n)
    gen_incorrect(run_name, args.p)

if __name__ == "__main__":
    main()
