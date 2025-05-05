import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import tempfile
import os
from PIL import Image
import pygame
import time

#pygame mixer for sound
pygame.mixer.init()

@st.cache_resource
def load_fire_model():
    return load_model("improved_fire_detection_model.h5")

model = load_fire_model()

def detect_fire(frame, threshold=0.5):
    preprocess_frame = cv2.cvtColor(cv2.resize(frame, (48, 48)), cv2.COLOR_BGR2GRAY)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=0)
    preprocess_frame = np.expand_dims(preprocess_frame, axis=-1)
    preprocess_frame = preprocess_frame.astype("float32") / 255
    
    prediction = model.predict(preprocess_frame)
    return prediction[0][1] >= threshold, prediction[0][1]

def trigger_alarm():
    try:
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play()
    except:
        st.warning("Could not play alarm sound. Check if alarm.mp3 exists.")

#Config UI
st.title("Fire Detection System")
st.write("Upload an image or video to detect fire")

tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

with tab1:
    st.header("Image Fire Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Converting PIL that is RGB image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        fire_detected, confidence = detect_fire(frame)
        
        if fire_detected:
            st.error(f"Fire detected! (Confidence: {confidence*100:.2f}%)")
            trigger_alarm()
            
            frame = cv2.rectangle(frame, (100, 100), (frame.shape[1] - 100, frame.shape[0] - 100), 
                                 (0, 0, 255), 2)
            frame = cv2.putText(frame, "Warning: FIRE DETECTED", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Converting back to RGB format for display
            result_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(result_image, caption="Fire Detection Result", use_column_width=True)
        else:
            st.success(f"No fire detected (Confidence: {(1-confidence)*100:.2f}%)")

with tab2:
    st.header("Video Fire Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_video is not None:
        # Saving uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.read())
            temp_video_path = tfile.name  

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()
        stop_button = st.button("Stop Video Processing")

        fire_detected_in_video = False
        last_alarm_time = 0
        alarm_cooldown = 5  

        try:
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                fire_detected, confidence = detect_fire(frame, threshold=0.4)

                # Convting to color for streamlit disp
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if fire_detected:
                    fire_detected_in_video = True
                    current_time = time.time()

                    frame_rgb = cv2.rectangle(
                        frame_rgb,
                        (100, 100),
                        (frame.shape[1] - 100, frame.shape[0] - 100),
                        (255, 0, 0),
                        2
                    )
                    frame_rgb = cv2.putText(
                        frame_rgb,
                        f"Fire Detected ({confidence * 100:.1f}%)",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # Trigger alarm 
                    if current_time - last_alarm_time > alarm_cooldown:
                        trigger_alarm()
                        last_alarm_time = current_time

                stframe.image(frame_rgb, channels="RGB")
                time.sleep(0.05)  

        finally:
            cap.release()
            try:
                os.unlink(temp_video_path)
            except PermissionError:
                st.warning(f"Could not delete temporary file: {temp_video_path}. It may still be in use.")

        if fire_detected_in_video:
            st.error("Fire was detected in the video!")
        else:
            st.success("No fire detected in the video.")

st.sidebar.header("About")
st.sidebar.info(
    """
    This Fire Detection System uses a Convolutional Neural Network (CNN) to detect fire in images and videos.
    - Upload an image or video file
    - The system will analyze it for fire
    - If fire is detected, an alarm will sound
    """
)