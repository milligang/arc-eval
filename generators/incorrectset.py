import argparse
import json
import os
import random
import shutil

# Example usage: python3 incorrectset.py --n 30 --p 10

def gen_random_set(n=20, base_path="../data/tasks/arc_agi_2"):
    """
    Populate random_set/ with N random tasks sampled from evaluation/.
    
    Args:
        n (int): Number of tasks to sample.
        base_path (str): Root ARC dataset path.
    """

    evaluation_path = os.path.join(base_path, "evaluation")
    random_set_path = os.path.join(base_path, "random_set")

    # Ensure evaluation directory exists
    if not os.path.isdir(evaluation_path):
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_path}")

    # Make sure random_set directory exists, and clear it
    if os.path.isdir(random_set_path):
        shutil.rmtree(random_set_path)
    os.makedirs(random_set_path)

    # Get all task folders
    all_tasks = [
        f for f in os.listdir(evaluation_path)
        if f.endswith(".json")
    ]

    if len(all_tasks) < n:
        raise ValueError(f"Requested {n} tasks but only {len(all_tasks)} available.")

    # Sample tasks
    sampled = random.sample(all_tasks, n)

    # Copy selected tasks into random_set
    for task in sampled:
        src = os.path.join(evaluation_path, task)
        dst = os.path.join(random_set_path, task)
        shutil.copyfile(src, dst)

def gen_incorrect_solutions(percentage=10, base_path="../data/tasks"):
    """
    Reads tasks from arc_agi_2/random_set and writes corrupted versions
    into data/tasks/incorrect_solutions.

    Args:
        percentage (float): Percentage of test output cells to corrupt.
        base_path (str): Root dataset path.
    """
    random_set_path = os.path.join(base_path, "arc_agi_2", "random_set")
    out_dir = os.path.join(base_path, "incorrect_solutions")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(random_set_path):
        raise FileNotFoundError(f"random_set directory not found: {random_set_path}")

    for fname in os.listdir(random_set_path):
        if not fname.endswith(".json"):
            continue

        in_path = os.path.join(random_set_path, fname)
        out_path = os.path.join(out_dir, fname)

        with open(in_path, "r") as f:
            data = json.load(f)

        # Corrupt only the test outputs
        for test_case in data.get("test", []):
            output_grid = test_case["output"]

            h = len(output_grid)
            w = len(output_grid[0])
            total_cells = h * w
            num_to_corrupt = int((percentage / 100.0) * total_cells)

            idxs = random.sample(range(total_cells), num_to_corrupt)

            for idx in idxs:
                r = idx // w
                c = idx % w
                output_grid[r][c] = random.randint(1, 9)

        # Save corrupted version
        with open(out_path, "w") as f:
            json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(
        description="Generate N random task solutions that are P-percent incorrect"
    )
    parser.add_argument(
        "--n", type=int, required=False,
        help="Number of tasks to sample"
    )
    parser.add_argument(
        "--p", type=int, required=False,
        help="Percentage to corrupt"
    )

    args = parser.parse_args()
    gen_random_set(args.n)
    gen_incorrect_solutions(args.p)

if __name__ == "__main__":
    main()