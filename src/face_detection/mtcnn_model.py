from __future__ import annotations

import numpy as np

from facenet_pytorch import MTCNN
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

from src.face_detection.base_face_detection  import BaseFaceDetection
from src.face_detection.image import Frame

if TYPE_CHECKING:
    from PIL.Image import Image

@dataclass
class MTCNNModel(BaseFaceDetection):

    mtcnn_kwargs: Optional[dict] = None

    def __post_init__(self):
        self.__model = MTCNN(**self.mtcnn_kwargs) if self.mtcnn_kwargs else MTCNN()

    def detect(self, image: Frame, offset_ratio: float = 0.05, return_frame = False) -> Union[Image, Frame]:
        """
        Detects faces in the image and returns the face as a PILLOW Image
        
        Parameters
        ----------
        image: src.face_detection.image.Frame
            Image to detect faces
        offset_ratio: float
            Ratio to add to the face bounding box to cover the whole head.

        Returns
        -------
        Image
            PILLOW Image of the face
        """
        img = image.get_pillow_image()
        boxes, probs, _ = self.__model.detect(img, landmarks=True)
        if isinstance(boxes, np.ndarray):
            image.update_face_bbox_location(*boxes[0])
            image.set_margin_ratio_to_face_bbox(ratio=offset_ratio)
            face_img = image.crop_face()
            if return_frame:
                return image
            return face_img
        else:
            if return_frame:
                return image
            return None