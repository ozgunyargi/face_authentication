import os, cv2, torch
import streamlit as st
from PIL import Image
from glob import glob
from typing import List

from config import mtcnn_params
from src import Frame, MTCNNModel
from src.face_recognition.authorization_database import AuthorizationDatabase
from src.face_recognition.inception_model import InceptionModel


@st.cache_resource
def face_detection() -> MTCNNModel:
    face_detection_model = MTCNNModel(mtcnn_params)

    return face_detection_model

@st.cache_resource
def extract_embeddings(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = InceptionModel(device=device)
    user_embedding = embedding_model.get_embedding(image)
    return user_embedding

def add_classroom():
    classrooms = [os.path.basename(room_path) for room_path in glob(f"{os.path.join(os.getcwd(), 'rooms')}/*")]
    st.subheader("Add Classroom")
    classroom_name = st.text_input("Enter the classroom name")
    if classroom_name not in classrooms and classroom_name != "":
        os.mkdir(os.path.join(os.getcwd(), "rooms", classroom_name))
        st.success(f"{classroom_name} is added successfully!")
    elif classroom_name != "":
        st.warning(f"{classroom_name} is already added.")

def add_user():
    st.subheader("Add User")
    classrooms = [os.path.basename(room_path) for room_path in glob(f"{os.path.join(os.getcwd(), 'rooms')}/*")]
    classroom = st.selectbox("Select Classroom", classrooms)
    username = st.text_input("Enter username")
    run = st.checkbox('Open Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    save_button = st.button('Save Image')

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pillow_image = Image.fromarray(frame)
        img = Frame(pillow_image)
        model = face_detection()
        face_img = model.detect(img, return_frame=True)
        img_w_bbox = face_img.visualize(return_image=True)
        FRAME_WINDOW.image(img_w_bbox)

        if save_button:
            ad = AuthorizationDatabase(room_path='rooms')
            face = model.detect(img)
            user_embedding = extract_embeddings(face)
            ad.add_user(room_name=classroom, user_id=username, embedding=user_embedding)
            st.balloons()  # Display balloons animation to indicate image is saved
            save_button = False
    else:
        st.write('Stopped')

def authenticate():
    ad = AuthorizationDatabase(room_path='rooms')
    st.subheader("Add User")
    classrooms = [os.path.basename(room_path) for room_path in glob(f"{os.path.join(os.getcwd(), 'rooms')}/*")]
    classroom = st.selectbox("Select Classroom", classrooms)
    run = st.checkbox('Open Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pillow_image = Image.fromarray(frame)
        img = Frame(pillow_image)
        model = face_detection()
        screen_img = model.detect(img, return_frame=True)
        face_img = model.detect(img)
        user_embedding = extract_embeddings(face_img)
        is_auth, _, _ = ad.authorize_user(room_name='FASSG042', new_embedding=user_embedding)
        if is_auth:
            border_color = "green"
        else:
            border_color = "red"
        img_w_bbox = screen_img.visualize(return_image=True, border_color= border_color)
        FRAME_WINDOW.image(img_w_bbox)

def app():
    st.title("Face Authentication for Classroom")

    options = ["add classroom", "add user", "authenticate"]
    option = st.sidebar.selectbox("Select an Option", options)

    if option == "add classroom":
        add_classroom()

    elif option == "add user":
        add_user()

    elif option == "authenticate":
        authenticate()



if __name__ == "__main__":
    app()