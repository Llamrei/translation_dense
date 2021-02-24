from datetime import datetime
from datetime import timedelta

class Timer:
    def __enter__(self):
        self._starts = dict()
        self.start('total')
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()

    @property
    def timing(self):
        results = dict()
        for name, start in self._starts.items():
            results[name] = (self.end - start)/timedelta(microseconds=1)
        return results

    def start(self, name):
        self._starts[f"time_{name}"] = datetime.now()