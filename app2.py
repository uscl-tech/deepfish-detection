import streamlit as st
import subprocess
import os
import pandas as pd
from PIL import Image
import supervision as sv
from inference import get_model


st.set_page_config(
    page_title="Underwater Sea Image detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    
    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = get_model(model_id="imageenhancement/2", api_key="eSzt9jqUwL3SzdHp0dmr")
    return model


def run_detection(name, checkpoint_dir, data_dir, epoch, test_name, result_dir, input_height, input_width):
    # Formulate the command to run the detection script
    detection_command = f"python test_funiegan.py --name {name} --checkpoint_dir {checkpoint_dir} --data_dir {data_dir} --epoch {epoch} --test_name {test_name} --result_dir {result_dir} --input_width {input_width} --input_height {input_height}"
    
    # Run the detection command using subprocess
    process = subprocess.Popen(detection_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check if detection ran successfully
    if process.returncode != 0:
        st.error(f"Error occurred during detection: {stderr.decode()}")
        return False
    else:
        return True


def calculate_psnr_ssim(oppath, data_dir, oppathcsv, resize, width, height):
    # Formulate the command to calculate PSNR and SSIM metrics
    psnr_ssim_command = f"python calc_psnr_ssim.py --input_dir {oppath} --refer_dir {data_dir} --output_dir {oppathcsv} {'--resize --width '+str(width)+' --height '+str(height) if resize else ''}"
    
    # Run the PSNR and SSIM calculation command using subprocess
    process = subprocess.Popen(psnr_ssim_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check if PSNR and SSIM calculation ran successfully
    if process.returncode != 0:
        st.error(f"Error occurred during PSNR/SSIM calculation: {stderr.decode()}")
        return None
    else:
        return os.path.join(oppathcsv, "quantitive_eval.csv")


def main():
    st.title("🌊 Underwater Sea Image Enhancement")
    
    # Sidebar for input parameters
    with st.sidebar:
        st.subheader("Detection Parameters")
        name = 'test'
        checkpoint_dir = 'pretrained'
        epoch = 95
        test_name = st.text_input("Test Name", "test1")
        result_dir = st.text_input("Result Directory", "results/funiegan01")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        input_height = st.number_input("Height", value=256)
        input_width = st.number_input("Width", value=512)

    if uploaded_file is not None:
        # Save the uploaded file to the data_dir
        data_dir = "Inputimages"
        os.makedirs(data_dir, exist_ok=True)
        image_path = os.path.join(data_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Image uploaded successfully: {uploaded_file.name}")
        
        # Display uploaded image
        st.subheader("Uploaded Image")
        uploaded_img = Image.open(image_path)
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)

        # Button to trigger detection and metrics calculation
        if st.button("Run Detection and Metrics"):
            if run_detection(name, checkpoint_dir, data_dir, epoch, test_name, result_dir, input_height, input_width):
                oppathcsv = os.path.join(result_dir, test_name)
                oppath = os.path.join(result_dir, test_name, "single", "predicted")
                metrics_file = calculate_psnr_ssim(
                    oppath, data_dir, oppathcsv, resize=True, width=256, height=256
                )
                
                if metrics_file is not None:
                    # Display output image if exists
                    output_path = os.path.join(result_dir, test_name, "single", "predicted", uploaded_file.name)
                    if os.path.exists(output_path):
                        st.subheader("Output Image")
                        output_img = Image.open(output_path)

                        model = load_model("temp")
                        res = model.infer(output_img, confidence=0.25)
                        detections = sv.Detections.from_inference(res[0].dict(by_alias=True, exclude_none=True))
                        
                        bounding_box_annotator = sv.RoundBoxAnnotator()
                        label_annotator = sv.LabelAnnotator()

                        # Annotate the image with inference results
                        annotated_image = bounding_box_annotator.annotate(scene=output_img, detections=detections)
                        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
                        st.image(annotated_image, caption="Detected Image", use_container_width=True)

                        # Check detected class names
                        class_names = detections.data['class_name']
                        for class_name in class_names:
                            st.text(class_name)
                            if class_name == "shark":
                                st.error("⚠️ Shark Ahead! Don't Proceed That Side!")
                            else:
                                st.success("✅ You Can Proceed Safely!")

                    else:
                        st.warning("⚠️ Output image not found. Please check the directory.")
                
                    # Display the calculated metrics (CSV file)
                    st.subheader("📊 PSNR and SSIM Metrics")
                    metrics_df = pd.read_csv(metrics_file)
                    st.write(metrics_df)


if __name__ == "__main__":
    main()
