import numpy as np
from file import get_predictions, get_arctask, find_line_by_uid
from arc.types import ArcIOPair
from agents import corrupt_grid, cmp_grids

# def hamming(arr1, arr2):
#     if arr1.shape != arr2.shape:
#         raise ValueError("Arrays must have the same shape.")
#     return np.sum(arr1 != arr2)

for j in range(0, 10):
    if (j == 2): continue
    task_id, ans = get_predictions(f"5pcrtxn{j}", "g25f0")
    last_ans = ans[-1]
    task_line = find_line_by_uid(task_id)
    task = get_arctask(task_line)

    if not cmp_grids(last_ans, task.test_pairs[0].y):        
        prediction = ArcIOPair(task.test_pairs[0].y, last_ans)
        prediction.plot(show=True, title=f"Task {j}")
    else:
        print(j)