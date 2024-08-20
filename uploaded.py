from ultralytics import YOLO
import streamlit as st
import tempfile
import io
import utils
import openvino as ov
import yaml
 

def load_OV(): 
    core = ov.Core()
    model = core.read_model(model = "uploaded_models/openvino.xml")
    compiled_model = core.compile_model(model = model, device_name = "AUTO")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    with open('models/metadata.yaml') as info:
        info_dict = yaml.load(info, Loader=yaml.Loader)

    labels = info_dict['names']

    return compiled_model, input_layer, output_layer
 
def load_PT(pt_model_name): 
    model_pt = YOLO(pt_model_name) 
  
    return model_pt

def run_page(conf_threshold):
    col1, col2 = st.columns(2)
    
 
    xml, bin, yaml, pytorch = None, None, None, None
    st.sidebar.header('OpenVINO')
    xml = st.sidebar.file_uploader("Upload an OpenVINO Model File (.xml).", type=(".xml"))
    bin = st.sidebar.file_uploader("Upload an OpenVINO Weights File (.bin).", type=( ".bin"))
    yaml = st.sidebar.file_uploader("Upload an OpenVINO YAML file (.metadata).", type=(".yaml"))
    st.sidebar.header('PyTorch')
    pytorch = st.sidebar.file_uploader("Upload a PyTorch Model file (.pt).", type=(".pt"))
    
    if None not in (xml, bin, yaml, pytorch ):
        pt_model_name = "uploaded_models/uploaded.pt"
        
        p = pytorch.getvalue()
        with open(pt_model_name, "wb") as f:
            f.write(p)

        x = xml.getvalue()
        with open("uploaded_models/openvino.xml", "wb") as f:
            f.write(x)

        b = bin.getvalue()
        with open("uploaded_models/openvino.bin", "wb") as f:
            f.write(b)

        y = yaml.getvalue()
        with open("uploaded_models/metadata.yaml", "wb") as f:
            f.write(y)
       
        with col1:
            st.header("OpenVINO")

            compiled_model, input_layer, output_layer = load_OV()
            utils.play_video_ov("assets/test.mp4", conf_threshold, compiled_model = compiled_model, input_layer = input_layer, output_layer = output_layer )

        with col2:
            
            st.header("Pytorch") 
            model_pt = load_PT(pt_model_name)
            utils.play_video_pt("assets/test.mp4", conf_threshold,  model_pt = model_pt)

    else: 
        st.write("Test the speed of your pytorch and OpenVINO models. Upload the models in the sidebar.")
 