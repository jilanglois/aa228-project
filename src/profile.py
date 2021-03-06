import time
from functools import wraps

PROF_DATA = {}


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        tot_time = sum(data[1])
        print("--------------- Function %s ----------------" % fname),
        print('TIME: Execution time max: %.3f, average: %.3f, total %.3f' % (max_time, avg_time, tot_time))
        print('COUNT: %d' % data[0])


def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}
