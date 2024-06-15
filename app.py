
# import streamlit as st
# import cv2
# import pandas as pd
# import datetime
# import os
# from PIL import Image

# # Define file paths for storing data
# attendance_file = "attendance/attendance_history.csv"
# user_data_file = "users/user_data.csv"

# # Ensure the directories for saved data exist
# os.makedirs("attendance", exist_ok=True)
# os.makedirs("users", exist_ok=True)

# # Function to load data from CSV files
# def load_data(file_path, columns):
#     if os.path.exists(file_path):
#         return pd.read_csv(file_path)
#     else:
#         return pd.DataFrame(columns=columns)

# # Initialize DataFrames from CSV files
# st.session_state.attendance_history = load_data(attendance_file, ["Name", "Email", "Date", "Time"])
# st.session_state.user_data = load_data(user_data_file, ["Name", "Email", "Image Path", "Date", "Time"])

# # Ensure the directory for saved images exists
# if not os.path.exists("registered_faces"):
#     os.makedirs("registered_faces")

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to load registered faces
# def load_registered_faces():
#     registered_faces = []
#     for root, dirs, files in os.walk("registered_faces"):
#         for file in files:
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 img_path = os.path.join(root, file)
#                 name = file.split("_")[0]
#                 email = file.split("_")[1].split(".")[0]
#                 registered_faces.append({"Name": name, "Email": email, "Image Path": img_path})
#     return registered_faces

# # Function to compare faces
# def compare_faces(face_crop, registered_faces):
#     for person in registered_faces:
#         registered_img = cv2.imread(person["Image Path"])
#         registered_img_gray = cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
#         face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
#         result = cv2.matchTemplate(face_crop_gray, registered_img_gray, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
#         if max_val > 0.8:
#             return person["Name"], person["Email"]
#     return None, None

# # Function to save data to CSV files
# def save_data(file_path, data):
#     data.to_csv(file_path, index=False)

# # Function to clear all data
# def clear_all_data():
#     if os.path.exists("attendance"):
#         for file in os.listdir("attendance"):
#             os.remove(os.path.join("attendance", file))
#     if os.path.exists("users"):
#         for file in os.listdir("users"):
#             os.remove(os.path.join("users", file))
#     if os.path.exists("registered_faces"):
#         for file in os.listdir("registered_faces"):
#             os.remove(os.path.join("registered_faces", file))
#     st.session_state.attendance_history = pd.DataFrame(columns=["Name", "Email", "Date", "Time"])
#     st.session_state.user_data = pd.DataFrame(columns=["Name", "Email", "Image Path", "Date", "Time"])

# st.sidebar.title("Face Attendance System")
# menu = st.sidebar.selectbox("Menu", ["Face Attendance", "Register Face", "History", "Clear Data"])

# if "start_camera" not in st.session_state:
#     st.session_state.start_camera = False

# if "captured_frame" not in st.session_state:
#     st.session_state.captured_frame = None

# if menu == "Face Attendance":
#     st.header("Face Attendance")
#     run = st.checkbox('Run', key="attendance_run")
#     FRAME_WINDOW = st.image([])
#     cap = cv2.VideoCapture(0)
#     registered_faces = load_registered_faces()

#     while run:
#         ret, frame = cap.read()
#         faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             face_crop = frame[y:y+h, x:x+w]
#             name, email = compare_faces(face_crop, registered_faces)
#             if name and email:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, f"Name: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 cv2.putText(frame, f"Email: {email}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 # Log attendance if not already recorded
#                 if not ((st.session_state.attendance_history['Name'] == name) & 
#                         (st.session_state.attendance_history['Email'] == email) & 
#                         (st.session_state.attendance_history['Date'] == datetime.date.today().strftime('%Y-%m-%d'))).any():
#                     now = datetime.datetime.now()
#                     new_entry = pd.DataFrame([{"Name": name, "Email": email, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
#                     st.session_state.attendance_history = pd.concat([st.session_state.attendance_history, new_entry], ignore_index=True)
#                     save_data(attendance_file, st.session_state.attendance_history)
#             else:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                 cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         FRAME_WINDOW.image(frame, channels="BGR")

#     cap.release()

# elif menu == "Register Face":
#     st.header("Register Face")

#     # Upload image section
#     image_file = st.file_uploader("Upload Image", type=["jpg", "png"], key="register_image")
#     if image_file is not None:
#         img = Image.open(image_file)
#         st.image(img, caption='Uploaded Image', use_column_width=True)
#         name = st.text_input("Name", key="upload_name")
#         email = st.text_input("Email", key="upload_email")
#         if st.button("Save", key="upload_save"):
#             st.checkbox("Are you want to save?")
#             img_path = os.path.join("registered_faces", f"{name}_{email}.jpg")
#             img.save(img_path)
#             now = datetime.datetime.now()
#             new_user = pd.DataFrame([{"Name": name, "Email": email, "Image Path": img_path, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
#             st.session_state.user_data = pd.concat([st.session_state.user_data, new_user], ignore_index=True)
#             save_data(user_data_file, st.session_state.user_data)
#             st.success("Face Registered Successfully and Image Saved")

#     st.write("OR")

#     # Capture from live video section
#     if st.button('Start Camera', key="start_camera_button"):
#         st.session_state.start_camera = True

#     if st.session_state.start_camera:
#         FRAME_WINDOW = st.image([])
#         cap = cv2.VideoCapture(0)
#         user = 1

#         while cap.isOpened():
#             ret, frame = cap.read()
#             faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#             FRAME_WINDOW.image(frame, channels="BGR")
#             if user<=1:
#                 user+=1
#                 if st.button("Capture", key="capture_button"):
#                     st.session_state.captured_frame = frame.copy()
#                     st.session_state.start_camera = False
#                     break

#         cap.release()

#     if st.session_state.captured_frame is not None:
#         st.image(st.session_state.captured_frame, caption='Captured Image', use_column_width=True)
#         name = st.text_input("Name", key="capture_name")
#         email = st.text_input("Email", key="capture_email")
#         if st.button("Save", key="capture_save"):
#             img_path = os.path.join("registered_faces", f"{name}_{email}.jpg")
#             cv2.imwrite(img_path, cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
#             now = datetime.datetime.now()
#             new_user = pd.DataFrame([{"Name": name, "Email": email, "Image Path": img_path, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
#             st.session_state.user_data = pd.concat([st.session_state.user_data, new_user], ignore_index=True)
#             save_data(user_data_file, st.session_state.user_data)
#             st.session_state.captured_frame = None  # Reset captured frame
#             st.success("Face Registered Successfully and Image Saved")

# elif menu == "History":
#     st.header("Attendance History")
#     st.dataframe(st.session_state.attendance_history)

#     st.header("Registered Users")
#     if not st.session_state.user_data.empty:
#         st.dataframe(st.session_state.user_data[["Name", "Email", "Date", "Time"]])
#     else:
#         st.write("No registered users found.")

#     # Load registered faces data
#     registered_faces = load_registered_faces()
#     if registered_faces:
#         registered_users_df = pd.DataFrame(registered_faces)
#         st.dataframe(registered_users_df[["Name", "Email"]])
#     else:
#         st.write("No registered users found.")

# elif menu == "Clear Data":
#     st.header("Clear All Data")
#     if st.button("Clear All Data", key="clear_data_button"):
#         clear_all_data()
#         st.success("All data has been cleared.")




import streamlit as st
import cv2
import pandas as pd
import datetime
import os
from PIL import Image

# Define file paths for storing data
attendance_file = "attendance/attendance_history.csv"
user_data_file = "users/user_data.csv"

# Ensure the directories for saved data exist
os.makedirs("attendance", exist_ok=True)
os.makedirs("users", exist_ok=True)

# Function to load data from CSV files
def load_data(file_path, columns):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=columns)

# Initialize DataFrames from CSV files
st.session_state.attendance_history = load_data(attendance_file, ["Name", "Email", "Date", "Time"])
st.session_state.user_data = load_data(user_data_file, ["Name", "Email", "Image Path", "Date", "Time"])

# Ensure the directory for saved images exists
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load registered faces
def load_registered_faces():
    registered_faces = []
    for root, dirs, files in os.walk("registered_faces"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                name = file.split("_")[0]
                email = file.split("_")[1].split(".")[0]
                registered_faces.append({"Name": name, "Email": email, "Image Path": img_path})
    return registered_faces

# Function to compare faces
def compare_faces(face_crop, registered_faces):
    for person in registered_faces:
        registered_img = cv2.imread(person["Image Path"])
        registered_img_gray = cv2.cvtColor(registered_img, cv2.COLOR_BGR2GRAY)
        face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(face_crop_gray, registered_img_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.8:
            return person["Name"], person["Email"]
    return None, None

# Function to save data to CSV files
def save_data(file_path, data):
    data.to_csv(file_path, index=False)

# Function to clear all data
def clear_all_data():
    if os.path.exists("attendance"):
        for file in os.listdir("attendance"):
            os.remove(os.path.join("attendance", file))
    if os.path.exists("users"):
        for file in os.listdir("users"):
            os.remove(os.path.join("users", file))
    if os.path.exists("registered_faces"):
        for file in os.listdir("registered_faces"):
            os.remove(os.path.join("registered_faces", file))
    st.session_state.attendance_history = pd.DataFrame(columns=["Name", "Email", "Date", "Time"])
    st.session_state.user_data = pd.DataFrame(columns=["Name", "Email", "Image Path", "Date", "Time"])

st.sidebar.title("Face Attendance System")
menu = st.sidebar.selectbox("Menu", ["Face Attendance", "Register Face", "History", "Clear Data"])

if "start_camera" not in st.session_state:
    st.session_state.start_camera = False

if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None

def start_camera():
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        st.warning(f"Cannot open camera with index ")
        return None
    return cap

if menu == "Face Attendance":
    st.header("Face Attendance")
    run = st.checkbox('Run', key="attendance_run")
    FRAME_WINDOW = st.image([])
    cap = start_camera()
    registered_faces = load_registered_faces()

    while run and cap:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture image from camera.")
            break

        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            name, email = compare_faces(face_crop, registered_faces)
            if name and email:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Name: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Email: {email}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # Log attendance if not already recorded
                if not ((st.session_state.attendance_history['Name'] == name) & 
                        (st.session_state.attendance_history['Email'] == email) & 
                        (st.session_state.attendance_history['Date'] == datetime.date.today().strftime('%Y-%m-%d'))).any():
                    now = datetime.datetime.now()
                    new_entry = pd.DataFrame([{"Name": name, "Email": email, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
                    st.session_state.attendance_history = pd.concat([st.session_state.attendance_history, new_entry], ignore_index=True)
                    save_data(attendance_file, st.session_state.attendance_history)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    if cap:
        cap.release()

elif menu == "Register Face":
    st.header("Register Face")

    # Upload image section
    image_file = st.file_uploader("Upload Image", type=["jpg", "png"], key="register_image")
    if image_file is not None:
        img = Image.open(image_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        name = st.text_input("Name", key="upload_name")
        email = st.text_input("Email", key="upload_email")
        if st.button("Save", key="upload_save"):
            st.checkbox("Are you sure you want to save?")
            img_path = os.path.join("registered_faces", f"{name}_{email}.jpg")
            img.save(img_path)
            now = datetime.datetime.now()
            new_user = pd.DataFrame([{"Name": name, "Email": email, "Image Path": img_path, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
            st.session_state.user_data = pd.concat([st.session_state.user_data, new_user], ignore_index=True)
            save_data(user_data_file, st.session_state.user_data)
            st.success("Face Registered Successfully and Image Saved")

    st.write("OR")

    # Capture from live video section
    if st.button('Start Camera', key="start_camera_button"):
        st.session_state.start_camera = True

    if st.session_state.start_camera:
        FRAME_WINDOW = st.image([])
        cap = start_camera()
        user = 1

        while cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image from camera.")
                break

            faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            FRAME_WINDOW.image(frame, channels="BGR")
            if user <= 1:
                user += 1
                if st.button("Capture", key="capture_button"):
                    st.session_state.captured_frame = frame.copy()
                    st.session_state.start_camera = False
                    break

        if cap:
            cap.release()

    if st.session_state.captured_frame is not None:
        st.image(st.session_state.captured_frame, caption='Captured Image', use_column_width=True)
        name = st.text_input("Name", key="capture_name")
        email = st.text_input("Email", key="capture_email")
        if st.button("Save", key="capture_save"):
            img_path = os.path.join("registered_faces", f"{name}_{email}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB))
            now = datetime.datetime.now()
            new_user = pd.DataFrame([{"Name": name, "Email": email, "Image Path": img_path, "Date": now.date().strftime('%Y-%m-%d'), "Time": now.time().strftime('%H:%M:%S')}])
            st.session_state.user_data = pd.concat([st.session_state.user_data, new_user], ignore_index=True)
            save_data(user_data_file, st.session_state.user_data)
            st.session_state.captured_frame = None  # Reset captured frame
            st.success("Face Registered Successfully and Image Saved")

elif menu == "History":
    st.header("Attendance History")
    st.dataframe(st.session_state.attendance_history)

    st.header("Registered Users")
    if not st.session_state.user_data.empty:
        st.dataframe(st.session_state.user_data[["Name", "Email", "Date", "Time"]])
    else:
        st.write("No registered users found.")

    # Load registered faces data
    registered_faces = load_registered_faces()
    if registered_faces:
        registered_users_df = pd.DataFrame(registered_faces)
        st.dataframe(registered_users_df[["Name", "Email"]])
    else:
        st.write("No registered users found.")

elif menu == "Clear Data":
    st.header("Clear All Data")
    if st.button("Clear All Data", key="clear_data_button"):
        clear_all_data()
        st.success("All data has been cleared.")
