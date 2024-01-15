import os, time
import cv2, torch
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
    if image is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = InceptionModel(device=device)
        user_embedding = embedding_model.get_embedding(image)
        return user_embedding
    else:
        return None

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
    save_button = st.button('Save Image')

    camera_opened = False

    while run:
        if not camera_opened and run:
            camera = cv2.VideoCapture(0)
            camera_opened = True
        _, frame = camera.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pillow_image = Image.fromarray(frame)
            img = Frame(pillow_image)
            model = face_detection()
            face_img = model.detect(img, return_frame=True)
            img_w_bbox = face_img.visualize(return_image=True)
            FRAME_WINDOW.image(img_w_bbox)

        if save_button:
            if face_img.bbox is not None:
                if len(username) > 0:
                    ad = AuthorizationDatabase(room_path='rooms')
                    face = model.detect(img)
                    user_embedding = extract_embeddings(face)
                    username = username.replace('_', '').replace(' ', '')
                    ad.add_user(room_name=classroom, user_id=username, embedding=user_embedding)
                    st.balloons()  # Display balloons animation to indicate image is saved
                    st.success(f"{username} is added to authorized users for {classroom}")
                    save_button = False
                else:
                    st.warning(f'Length of username should be more than 1 character!')
                    save_button = False
            else:
                st.warning(f'Face is not detected. Please wait for the red box before saving.')
                save_button = False
    

def remove_user():
    st.subheader("Remove User")
    ad = AuthorizationDatabase(room_path='rooms')
    # List all rooms
    classrooms = [os.path.basename(room_path) for room_path in glob(f"{os.path.join(os.getcwd(), 'rooms')}/*")]
    selected_room = st.selectbox("Select Classroom", classrooms)
    # Get authorized users for the selected room
    authorized_users = [file.split('_')[0] for file in os.listdir(os.path.join('rooms', selected_room)) if file.endswith('_emb.pt')]
    # Show authorized users
    selected_user = st.selectbox("Select User to Remove", authorized_users)

    if st.button("Remove User"):
        if selected_user is not None:
            ad.remove_user(room_name=selected_room, user_id=selected_user)
            st.success(f"User '{selected_user}' removed from '{selected_room}'")
            st.rerun()


def authenticate():
    ad = AuthorizationDatabase(room_path='rooms')
    st.subheader("Authenticate")
    classrooms = [os.path.basename(room_path) for room_path in glob(f"{os.path.join(os.getcwd(), 'rooms')}/*")]
    classroom = st.selectbox("Select Classroom", classrooms)
    run = st.checkbox('Open Camera')
    FRAME_WINDOW = st.image([])
    
    border_color = "red"
    max_failures = 10
    failure_count = 0
    authentication_done = False
    camera_opened = False

    while run:
        if not camera_opened and run:
            camera = cv2.VideoCapture(0)
            camera_opened = True
            st.info("Camera Starts")
            st.info("Authorizing User..")

        _, frame = camera.read()
        if frame is not None:
            # Screen and bbox capture
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pillow_image = Image.fromarray(frame)
            img = Frame(pillow_image)
            model = face_detection()
            screen_img = model.detect(img, return_frame=True)
            # Detection and Recognition
            if not authentication_done:
                face_img = model.detect(img)
                user_embedding = extract_embeddings(face_img)
            
            if user_embedding is not None and not authentication_done:
                is_auth, _, _ = ad.authorize_user(room_name=classroom, new_embedding=user_embedding)
                if is_auth:
                    border_color = "green"
                    st.success("Authentication successful. Access granted!")
                    authentication_done = True
                else:
                    border_color = "red"
                    failure_count += 1
                    if failure_count >= max_failures:
                        st.error(f"Authentication failed. Access denied.")
                        camera.release()
                        st.error("Camera stopped.")
                        break
            img_w_bbox = screen_img.visualize(return_image=True, border_color= border_color)
            FRAME_WINDOW.image(img_w_bbox)
            
            if screen_img.bbox is None and authentication_done:
                time.sleep(2)
                camera.release()
                st.error("Detection lost. Camera stopped.")



def app():
    st.title("Face Authentication for Classroom")

    options = ["Authenticate", "Add Classroom", "Add User", "Remove User"]
    option = st.sidebar.selectbox("Select an Option", options)

    if option == "Authenticate":
        authenticate()
    elif option == "Add Classroom":
        add_classroom()
    elif option == "Add User":
        add_user()
    elif option == "Remove User":
        remove_user()



if __name__ == "__main__":
    app()