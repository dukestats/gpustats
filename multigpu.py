from threading import Thread

import testmod

class GPUCall(Thread):
    """

    """

    def __init__(self, func, device=0):
        self.func = func
        self.device = device

    def acquire_device(self):
        testmod.set_device(self.device)

    def release_device(self):
        pass

    def run(self):
        self.acquire_device()
        self.func()
        self.release_device()

def make_calls(func, data, devices=None, splits=None):
    """

    Parameters
    ----------

    Returns
    -------

    """
    if splits is None:
        pass

def _execute_calls(calls):
    """

    """
    for call in calls:
        call.start()

    for call in calls:
        call.join()




