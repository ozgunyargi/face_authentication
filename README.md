# Face Authentication

Özgün Yargı 20811

Mert Pekey 23646

## Introduction

In this project, we developed a web application that authenticates people through a camera for given classrooms. We chose a web application because it can easily scale up when adding another room to the system. Each embedded system can connect to the website, set up its authentication, and integrate with the application database. By using a web application, we can access a variety of hardware sources via cloud, allowing us to employ more powerful deep learning models. This capability is essential for effective and real time face detection and recognition.

Our web application consists of four functions. First function is called “Authenticate”. This function uses camera frames to detect whether a person is eligible to enter the classroom or not. If it is eligible, a green bounding box with the id name of it will be displayed in camera. If it has no access, a red bounding box will be displayed. For different classroom authentication lists, you may change the classroom. Next function is “Add Classroom” which basically adds a new classroom to the database. Another functionality of the web application is “Add User” which introduces the facial features of the introduced user for the selected classroom. This function uses a camera to detect the facial features and after the user presses the “Save Image” button, the detected face is cropped from the frame and embedding of the face is extracted to be later stored for a chosen classroom. The last functionality is called “Remove User” which removes previously authenticated user from the selected classroom.

## Installation

You can install the required libraries using the following pip commands:

```bash
pip install -r requirements.txt
```

## Running Streamlit Application

```bash
streamlit run streamlit_app.py
```

If the application does not automatically open in your web browser, you can manually access it by opening your browser and navigating to:
http://localhost:8501/