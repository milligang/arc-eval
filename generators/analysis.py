import numpy as np
from file import get_predictions, get_arctask, find_line_by_uid
from arc.types import ArcIOPair

def hamming(arr1, arr2):
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape.")
    return np.sum(arr1 != arr2)

task_id, ans = get_predictions("2crtxn6", "g25f1")
last_ans = ans[-1]
task_line = find_line_by_uid(task_id)
task = get_arctask(task_line)

print(len(ans))
# ham = hamming(last_ans, task.test_pairs[0].y)
ham = 1
if ham != 0:
    print(ham)
    # for i, pair in enumerate(task.train_pairs, start=1):
    #     pair.plot(show=True, title=f"Demo {i}")

    for i, pair in enumerate(task.test_pairs, start=1):
        pair.plot(show=True, title=f"Test {i}")
    
    for p in ans:
        prediction = ArcIOPair(task.test_pairs[0].y, p)
        prediction.plot(show=True)