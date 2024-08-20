import utils
import uploaded
#import utils_compare
import time
import streamlit as st
import PIL
import numpy as np
import io
from PIL import Image
import tempfile
import moviepy.editor as mpy
from camera_input_live import camera_input_live

import cv2

st.set_page_config(
    page_title = "AI Kickboard Safety Project", 
    page_icon = ":scooter:",
    layout = "centered", 
    initial_sidebar_state = "expanded")

st.title("AI Kickboard Safety Project :scooter:")




st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM", "COMPARE", "UPLOAD"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the confidence threshold", 0, 100, 20))/100

input = None 
temporary_location = None

input_compare = None 
temporary_location_compare = None
if source_radio == "COMPARE":

    st.sidebar.header("Upload")
    input_compare = st.sidebar.file_uploader("Choose a video.", type=("mp4"))
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("OpenVINO")
 
    
    with col2:
        st.header("Pytorch")
   

    if input_compare is not None:
 
        g = io.BytesIO(input_compare.read())
        temporary_location_compare = "upload_compare.mp4" 

        with open(temporary_location_compare, "wb") as out: 
            out.write(g.read())

        out.close() 

    if temporary_location_compare is not None:
        col1, col2 = st.columns(2)

        with col1:
            utils.play_video_ov(temporary_location_compare, conf_threshold)
        
        with col2:
            utils.play_video_pt(temporary_location_compare, conf_threshold)

    else:
        #st.video("assets/sample_video.mp4")
        st.write("Upload a video in the sidebar to compare the speeds between a OpenVINO model and PyTorch model." )

if source_radio == "UPLOAD":

    uploaded.run_page(conf_threshold)

if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        
        visualized_image, inference_time = utils.predict_image(uploaded_image_cv, conf_threshold)
         
        st.image(visualized_image, channels = "BGR")

    else: 
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image." )
        st.image("assets/sample_image.jpg")
    


if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))

    if input is not None:

        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4" 

        with open(temporary_location, "wb") as out: 
            out.write(g.read())

        out.close() 

    if temporary_location is not None:

        utils.play_video(temporary_location, conf_threshold)

    else:
        st.write("Click on 'Browse Files' in the sidebar to run inference on an video." )
        st.video("assets/sample_video.mp4")
        


if source_radio == "WEBCAM":
    image = camera_input_live()

    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image, inference_time = utils.predict_image(uploaded_image_cv, conf_threshold)

    st.image(visualized_image, channels = "BGR")
 





















