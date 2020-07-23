
from timeit import default_timer


class MeasureTime:
    """
    Easily measure the time of a Python code block.
    Example:
        with MeasureTime():
            # Do stuff
            # Measured time is automatically printed to the console
    """
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.start = default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        end = default_timer()
        seconds = end - self.start

        if self.name:
            tag = '[' + self.name + '] '
        else:
            tag = ''

        print('%sElapsed time: %d m and %.2f s' % (tag, seconds // 60, seconds % 60))
