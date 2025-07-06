class Log:
    def __init__(self):
        self._in_debug = False

    def debug_mode(self, in_debug: bool):
        self._in_debug = in_debug

    def __call__(self, *info):
        if self._in_debug:
            print(info)


log = Log()
