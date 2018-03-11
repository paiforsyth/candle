import time
import datetime
from time import localtime, strftime
import math
import logging



def as_minutes(s):
    m = math.floor(s / 60)
    secs = s - m * 60
    return '%dm %ds' % (m, secs)


def time_since(since, progress):
    now = time.time()
    s = now - since
    if progress == 0:
        return asMinutes(s), "unknown"
    es = s / (progress)
    rs = es - s
    return as_minutes(s), as_minutes(rs)

def progress_info(starttime, curiter, totaliter):
    progress= curiter/totaliter
    elapsed,estimated_remaining=time_since(starttime, progress)
    return (progress, elapsed, estimated_remaining)



def report( starttime, curiter, totaliter, loss):
        progress, elapsed, est = progress_info(starttime, curiter, totaliter)
        s = "Elapsed time: %s. Iteration: %d. Progress: %d. Remaining: %s. loss: %.6f " % (
            elapsed, curiter, 100 * progress, est, loss)
        logging.info(s)

def timestamp():
    return datetime.datetime.now().strftime("%d_%B_%Y_%A_%H_%M_%S")

def daystamp():
    return datetime.datetime.now().strftime("%d_%B_%Y_%A")

