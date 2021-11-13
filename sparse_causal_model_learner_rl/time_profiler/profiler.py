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
        self.shorten_name = True
        self.sort_children = True

    def delta(self):
        return self.t_end - self.t_start

    def print_delta(self):
        delta = self.delta()
        if delta >= 1.0:
            return '%.2f' % delta + 's'
        elif delta >= 0.001:
            return '%.2f' % (delta * 1000) + 'ms'
        elif delta >= 0.000001:
            return '%.2f' % (delta * 1000000) + 'mcs'
        else: return str(round(delta, 10))

    def describe(self, offset=0):
        if self.parent:
            percent = self.delta() / self.parent.delta()
        else:
            percent = 1.0
        percent = round(percent * 100, 2)
        indent = "|   " * offset
        accounted_children_time = round(sum([x.delta() for x in self.children]) / self.delta() * 100, 2)
        name = self.name
        if self.parent and self.shorten_name:
            assert name.startswith(self.parent.name)
            name = name[len(self.parent.name) + 1:]
        print(f"{indent}- {name} {percent}% // {self.print_delta()} [accounted {accounted_children_time}%]")
        children = self.children
        if self.sort_children:
            children = sorted(children, key=lambda c: -c.delta())
        for item in children:
            item.describe(offset=offset + 1)

    def __repr__(self):
        return self.name


@gin.configurable
class TimeProfiler(object):
    """Profile execution."""
    def __init__(self, enable=False, strict=False):
        self.time_start = {}
        self.time_end = {}
        self.events = []
        self.enable = enable
        self.prefix = ''
        self.prefixes = []
        self.strict = strict
        self.start('profiler')

    def error(self, msg):
        if self.strict:
            raise ValueError(msg)
        else:
            logging.error(msg)

    def _set_prefix(self, p):
        self.prefixes.append(p)
        self.prefix = self._prefix

    def _pop_prefix(self, p):
        if self.prefixes[-1] != p:
            msg = f"pop_prefix() value is invalid. Existing: {self.prefixes}, got {p}"
            self.error(msg)
        self.prefixes = self.prefixes[:-1]
        self.prefix = self._prefix

    @property
    def _prefix(self):
        return ''.join([p + '_' for p in self.prefixes])

    def start(self, name):
        self._start(name)
        self._set_prefix(name)
    
    def end(self, name):
        self._pop_prefix(name)
        self._end(name)

    def _start(self, name):
        name = self.prefix + name
        if name in self.time_start:
            self.error(f"{name} already started")
        t = time()
        self.time_start[name] = t
        self.events.append(('start', name, t))

    def _end(self, name):
        name = self.prefix + name
        if name not in self.time_start:
            msg = f"{name} was not started yet, started items: {list(self.time_start.keys())}"
            self.error(msg)
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
        self.end('profiler')
        if not self.enable:
            return
        print("==== PROFILE ====")
        self.nested().describe()
        print("\n\n")

