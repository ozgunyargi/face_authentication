from config import mtcnn_params

from src import MTCNNModel
from src import Frame

def main():
    image = Frame(image_path)
    detected_face_image = MTCNNModel(mtcnn_kwargs=mtcnn_params).detect(image)
    face_frame = Frame(img=detected_face_image)
    face_frame.visualize()

if __name__ == "__main__":
    image_path = r"D:\Users\suuser\Pictures\Saved Pictures\pp.jpg"
    main()