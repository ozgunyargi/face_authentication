from abc import ABC, abstractmethod
from src.face_detection.image import Frame

class BaseFaceDetection(ABC):

    @abstractmethod
    def detect(self, image: Frame):
        pass