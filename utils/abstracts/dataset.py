from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_loader(self):
        pass

    @abstractmethod
    def render(self):
        pass

