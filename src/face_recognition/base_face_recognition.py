from abc import ABC, abstractmethod
from PIL.Image import Image

class BaseEmbeddingModel(ABC):

    @abstractmethod
    def get_embedding(self, image: Image):
        pass