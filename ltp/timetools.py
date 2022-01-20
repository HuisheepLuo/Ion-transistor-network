import time
from functools import wraps

def timefn(fn):
    """
    I'm a timer.
    """
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"{fn.__name__} took {t2 - t1: .5f} s.")
        return result
    return measure_time 