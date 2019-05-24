"""Module with implementations of heaps."""
from typing import NamedTuple, List, Optional, Iterable

from job import Job


class HeapObject(NamedTuple):
    job: Job
    priority: int

    def __eq__(self, other):
        return self.priority == other.priority

    def __gt__(self, other):
        if self.priority == other.priority:
            return self.job.id > other.job.id
        return self.priority > other.priority

    def __lt__(self, other):
        return self.priority < other.priority


class Heap:
    """Heap with minimal preparation time at its root."""
    def __init__(self, initial: Optional[List[HeapObject]] = None) -> None:
        self._heap = []
        if initial is not None:
            for item in initial:
                self.push(item.job, item.priority)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self):
        yield from self._heap

    def push(self, job: Job, priority: int) -> None:
        """Push new item to the heap."""
        item = HeapObject(job, priority)
        self._heap.append(item)
        self._heapify_up()

    def extend(self, iterable: Iterable):
        """Extend heap with iterable."""
        for job, priority in iterable:
            self.push(job, priority)

    def peek(self) -> Job:
        """Return root of a heap without popping it."""
        return self._heap[0].job

    def pop(self) -> Job:
        """Pop root of the heap and return it."""
        last_item = self._heap.pop()
        if self._heap:
            return_item = self._heap[0]
            self._heap[0] = last_item
            self._heapify_down()
            return return_item.job
        return last_item.job

    @staticmethod
    def _parent_index(index):
        return (index - 1) // 2

    @staticmethod
    def _left_child_index(index):
        return 2 * index + 1

    @staticmethod
    def _right_child_index(index):
        return 2 * index + 2

    def _has_parent(self, index):
        return self._parent_index(index) >= 0

    def _has_left_child(self, index):
        return self._left_child_index(index) < len(self._heap)

    def _has_right_child(self, index):
        return self._right_child_index(index) < len(self._heap)

    def _parent(self, index) -> HeapObject:
        return self._heap[self._parent_index(index)]

    def _left_child(self, index):
        return self._heap[self._left_child_index(index)]

    def _right_child(self, index):
        return self._heap[self._right_child_index(index)]

    def _heapify_up(self):
        """Restore heap property from bottom to top."""
        index = len(self._heap) - 1
        while self._has_parent(index) and self._parent(index) > self._heap[index]:
            parent_index = self._parent_index(index)
            self._heap[parent_index], self._heap[index] = (
                self._heap[index], self._heap[parent_index]
            )
            index = parent_index

    def _heapify_down(self):
        """Restore heap property from top to bottom."""
        index = 0
        while self._has_left_child(index):
            child_index = self._left_child_index(index)
            right_index = child_index + 1
            if (right_index < len(self._heap)
                    and self._heap[right_index] < self._heap[child_index]):
                child_index = right_index
            if self._heap[index] < self._heap[child_index]:
                break
            else:
                self._heap[index], self._heap[child_index] = (
                    self._heap[child_index], self._heap[index]
                )
            index = child_index
