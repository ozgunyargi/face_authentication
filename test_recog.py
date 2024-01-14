import torch

from config import mtcnn_params
from src.face_detection.image import Frame
from src.face_detection.mtcnn_model import MTCNNModel

from src.face_recognition.inception_model import InceptionModel
from src.face_recognition.authorization_database import AuthorizationDatabase

def main(image):
    # Get embedding of a new user
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = InceptionModel(device=device)
    user_embedding = embedding_model.get_embedding(image)

    # Check whether the user is authorized for the room
    ad = AuthorizationDatabase(room_path='rooms')
    #ad.create_room(room_name='FENS2014')
    #ad.add_user(room_name='FENS2014', user_id='MertPekey', embedding=user_embedding)
    is_auth, auth_user, auth_similarity = ad.authorize_user(room_name='FENS2014', new_embedding=user_embedding)
    if is_auth:
        print(f'{auth_user} is authorized to enter the room. There is {auth_similarity}% similarity')
    else:
        if auth_similarity:
            print(f'User is not authorized to enter the room. Max similarity is {auth_similarity}')
        print('User is not authorized to enter the room')

if __name__ == "__main__":
    image_path = '/Users/mpekey/Desktop/face_authentication/src/face_recognition/deneme_foto3.jpg'
    image = Frame(image_path)
    detected_face_image = MTCNNModel(mtcnn_kwargs=mtcnn_params).detect(image)
    face_frame = Frame(img=detected_face_image)
    main(face_frame.img)