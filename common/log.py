import os

from datetime import datetime


def prepare_directory(name):
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    logdir = os.path.join('logs', name + '_' + date)
    if os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
