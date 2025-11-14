from abc import ABC, abstractmethod


class Visualizer(ABC):

    def __init__(self, data, plot_type):
        self.data = data
        self.plot_type = plot_type

    def draw(self):
        raise NotImplementedError("This method must be implemented")
