import os

from datetime import datetime
from nnabla.monitor import Monitor


def prepare_monitor(name):
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    logdir = os.path.join('logs', name + '_' + date)
    if os.path.exists(logdir):
        os.makedirs(logdir)
    return Monitor(logdir)
