from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from src.face_recognition.base_face_recognition import BaseEmbeddingModel

class InceptionModel(BaseEmbeddingModel):
    def __init__(self, device='cpu'):
        self.device = device
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def get_embedding(self, image):
        """
        Retrieves the embedding of a given PIL Image.

        Parameters
        ----------
        image: PIL.Image.Image
            The input image for which the embedding is to be obtained.

        Returns
        -------
        torch.Tensor
            The embedding of the input image.
        """
        image = self.transform(image).unsqueeze(0).to(self.device)
        return self.facenet(image)