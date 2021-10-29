import gin
import logging
from collections import deque
from time import time


class ProfilerItem:
    """Store one measurable item with children/parent."""
    def __init__(self, name, t_start, t_end, parent=None, children=None):
        if children is None:
            children = []
        self.name = name
        self.t_start = t_start
        self.t_end = t_end
        self.parent = parent
        self.children = children

    def delta(self):
        return self.t_end - self.t_start

    def describe(self, offset=0):
        if self.parent:
            percent = self.delta() / self.parent.delta()
        else:
            percent = 1.0
        percent = round(percent * 100, 2)
        indent = "    " * offset
        print(f"{indent}- {self.name} {percent}% of {self.parent}")
        for item in self.children:
            item.describe(offset=offset + 1)

    def __repr__(self):
        return self.name


@gin.configurable
class TimeProfiler(object):
    """Profile execution."""
    def __init__(self, enable=False):
        self.time_start = {}
        self.time_end = {}
        self.events = []
        self.enable = enable
        self.start('profiler')

    def start(self, name):
        t = time()
        self.time_start[name] = t
        self.events.append(('start', name, t))

    def end(self, name):
        t = time()
        self.time_end[name] = t
        self.events.append(('end', name, t))

    def delta(self, name):
        return self.time_end.get(name, 0) - self.time_start.get(name, 0)

    def nested(self):
        # format: list of items, each is potentially a list of items
        # item -> [children], duration, name
        assert sorted(self.time_start.keys()) == sorted(self.time_end.keys()),\
            f"Some keys are different in start/end: {self.time_start} {self.time_end}"
        events_sorted = sorted(self.events, key=lambda x: x[2])
        # print(events_sorted)
        current_item = None
        root_item = None
        for act, name, t in events_sorted:
            if act == 'start':
                new_item = ProfilerItem(name=name, t_start=t, t_end=None, parent=current_item)
                if current_item:
                    current_item.children.append(new_item)
                else:
                    root_item = new_item
                current_item = new_item
            elif act == 'end':
                if current_item is None or name != current_item.name:
                    logging.error("Check your start/end commands")
                    continue
                current_item.t_end = t
                current_item = current_item.parent

        return root_item

    def report(self):
        if not self.enable:
            return
        self.end('profiler')
        print("==== PROFILE ====")
        self.nested().describe()
        print("\n\n")

