from dataclasses import dataclass
from typing import Union, Optional
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Frame:
    """
    Class to represent an image frame and its face bounding box location

    Attributes
    ----------
    img: Union[Image.Image, str]
        Image to detect faces
    width: int
        Width of the image
    height: int
        Height of the image
    x1: int
        X coordinate of the bottom left corner of the image
    y1: int
        Y coordinate of the bottom left corner of the image
    x2: int
        X coordinate of the top right corner of the image
    y2: int
        Y coordinate of the top right corner of the image
    face_x1: int
        X coordinate of the bottom left corner of the face bounding box
    face_y1: int
        Y coordinate of the bottom left corner of the face bounding box
    face_x2: int
        X coordinate of the top right corner of the face bounding box
    face_y2: int
        Y coordinate of the top right corner of the face bounding box
    bbox: list
        List of the face bounding box coordinates [x1, y1, x2, y2]
    """
    img: Union[Image.Image, str] = None

    def __post_init__(self):

        if not self.img:
            raise ValueError('Image is required')

        if isinstance(self.img, str):
            self.img = Image.open(self.img)

        self.width, self.height = self.img.size
        self.x1, self.y1, self.x2, self.y2 = 0, 0, self.width, self.height

        self.face_x1, self.face_y1 = None, None
        self.face_x2, self.face_y2 = None, None
        self.bbox = None

    def get_pillow_image(self) -> Image.Image:
        """
        Returns the image as a PILLOW Image
        """
        return self.img
    
    def update_face_bbox_location(self, x1, y1, x2, y2):
        """
        Updates the face bounding box location

        Parameters
        ----------
        x1: int
            X coordinate of the bottom left corner of the face bounding box
        y1: int
            Y coordinate of the bottom left corner of the face bounding box
        x2: int
            X coordinate of the top right corner of the face bounding box
        y2: int
            Y coordinate of the top right corner of the face bounding box
        """
        self.face_x1, self.face_y1 = x1, y1
        self.face_x2, self.face_y2 = x2, y2
        self.bbox = [x1, y1, x2, y2]

    def set_margin_ratio_to_face_bbox(self, ratio: float = 0.2) -> None:
        """
        Sets the margin ratio to the face bounding box to cover more area around the face bounding box

        Parameters
        ----------
        ratio: float
            Ratio to add to the face bounding box to cover the whole head. Ratio is defined as a percentage of the face bounding box width and height.
        """
        width_ratio = self.width * ratio
        height_ratio = self.height * ratio

        self.face_x1, self.face_x2 = self.face_x1 - width_ratio/2, self.face_x2 + width_ratio/2
        self.face_y1, self.face_y2 = self.face_y1 - height_ratio/2, self.face_y2 + height_ratio/2

    def crop_face(self) -> Image.Image:
        """
        Crops the face from the image and returns the face as a PILLOW Image
        """
        return self.img.crop((self.face_x1, self.face_y1, self.face_x2, self.face_y2))

    def visualize(self, border_color: str = "red", save_path: Optional[str]=None, return_image: bool = False, text: Optional[str] = None):
        """
        Visualizes the face bounding box on the image and saves it to the given path if provided.

        Parameters
        ----------
        save_path: Optional[str]
            Path to save the image with the face bounding box drawn on it.
        """
        img_to_draw = self.img.copy()
        draw = ImageDraw.Draw(img_to_draw)

        if self.face_x1:
            draw.rectangle(((self.face_x1, self.face_y1), (self.face_x2, self.face_y2)), outline=border_color, width=3)
            if text:
                fontsize = 24
                font = ImageFont.truetype("arial.ttf", fontsize)
                ratio = 0.06
                draw.text(xy=(self.face_x1, self.face_y1-self.height*ratio), text=text, fill=border_color, font=font)
        if save_path:
            img_to_draw.save(save_path)
        if return_image:
            return img_to_draw
        else:
            img_to_draw.show()