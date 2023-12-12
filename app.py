import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gc
import io
import warnings
from PIL import Image
from tensorflow.keras.metrics import top_k_categorical_accuracy
from skimage.metrics import structural_similarity
import cv2



img_1_path = "assets/Image1.jpg"
img_2_path = "assets/Image2.jpg"
img_3_path = "assets/Image3.jpg"

MODEL_PATH = 'model/optimisedmodel121.h5'
LABELS_PATH = 'assets/modelclasses.txt'

def gather_patient_info():
    st.sidebar.markdown("**Patient Information**")
    patient_id = st.sidebar.text_input("Patient ID:")
    age = st.sidebar.slider("Patient Age:", 1, 100, 25)
    gender = st.sidebar.radio("Patient Gender:", ["Male", "Female", "Other"])
    anatomical_site = st.sidebar.selectbox("Anatomical Site:", ["Head/Neck", "Upper Limb", "Lower Limb", "Trunk", "Genital", "Other"])

    # Store patient information in a dictionary
    patient_info = {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Anatomical Site": anatomical_site
    }

    return patient_info


def is_relevant_image(test_image):
    base_image = cv2.imread('assets/Image1.jpg')

    test_image_resized = cv2.resize(test_image, (base_image.shape[1], base_image.shape[0]))
    # Convert images to grayscale
    
    image1_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity index (SSIM)
    ssim_score = structural_similarity(image1_gray, image2_gray)
    similarity_threshold = 0.56

    return ssim_score > similarity_threshold

# Define custom metrics
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def load_image():
    uploaded_file = st.file_uploader(label='Upload an Image in the format below', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try: 
            image_data = uploaded_file.read()
            test_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            st.image(image_data, caption='Uploaded Image.', use_column_width=True)

            if is_relevant_image(test_image):
                st.success("Your image is relevant!")

            else:
                st.error("The uploaded image is not relevant to skin cancer.")
                return None
                
            return Image.open(io.BytesIO(image_data)).convert('RGB')

        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            return None

    else:
        return None

def load_existing_image(image_path):
    st.write("**Selected Image for DEMO**")
    st.image(image_path, caption='Selected Image.', use_column_width=True)
    return Image.open(image_path).convert('RGB')

def load_existing():
    st.write("**Select an image for a DEMO**")
    menu = ["Select an Image", "Image 1", "Image 2", "Image 3"]
    choice = st.selectbox("Select an image", menu)
    if choice == "Image 1":
        return load_existing_image(img_1_path)
    elif choice == "Image 2":
        return load_existing_image(img_2_path)
    elif choice == "Image 3":
        return load_existing_image(img_3_path)
    else:
        return None

def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

def preprocess_image(test_image, img_size=(224, 224)):
    img = test_image.resize(img_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, label_mapping, categories, image, patient_info=None, selected_choice=None):
    img_array = preprocess_image(image)
    
    with st.spinner('Classifying...'):
        predictions = model.predict(img_array)

    probabilities = predictions[0]

    all_prob, all_catid = tf.math.top_k(probabilities, k=4)
    all_prob = all_prob.numpy()
    all_catid = all_catid.numpy()

    # Map short-form labels to long-form labels
    long_labels = [label_mapping[categories[cat_id]] for cat_id in all_catid]

    # Create a DataFrame for the predictions
    data = {'Classes': long_labels,
            'Confidence %': all_prob}
    df_predictions = pd.DataFrame(data)

    st.subheader('Top Predictions:')
    # Get the top prediction
    top_prediction_label = long_labels[0]
    top_prediction_confidence = all_prob[0]

    st.write(f"Here are the top 4 predictions. The predicted type of skin cancer is **{top_prediction_label}** with a confidence level of **{top_prediction_confidence:.2%}**.")

    st.dataframe(df_predictions, use_container_width=True)
        
    if selected_choice == "Upload Image":
        download_patient_data(df_predictions)
            
    elif selected_choice == "Select Existing Image":
        pass   
    
     
    st.info(
        """
        :warning: **Disclaimer:**
        DermDetect AI is a tool designed to assist in the classification of skin cancer images, but it is not a substitute for professional medical advice, diagnosis, or treatment. The predictions made by the AI can be incorrect, and users are strongly advised to cross-check results with the guidance of a qualified healthcare professional.

        Always consult with a medical professional for accurate and personalized medical advice.

        """
    )
    
    

def download_patient_data(df_predictions):
    # Create a DataFrame for patient information
    df_patient_info = pd.DataFrame(st.session_state.patient_info, index=[0])
    patient_id = st.session_state.patient_info.get("Patient ID", "Unknown")

    # Combine patient information with predictions
    df_combined = pd.concat([df_patient_info, df_predictions], axis=1)
    
    # Save DataFrame to CSV and download
    csv_data = df_combined.to_csv(index=False, encoding='utf-8-sig')
    filename = f"patient_data_{patient_id}.csv"

    st.download_button(
        label="Download Patient Data",
        data=csv_data,
        file_name=filename,
        key="download_button"
    )
     # Clear the session state after downloading data
    st.session_state.clear()
    
def main():
    gc.enable()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    st.set_page_config(page_title='DermDetect', page_icon=':microscope:')

    title_column, credit_column = st.columns([2, 1])

    # Title Section
    with title_column:
        st.title('DermDetect AI :microscope:')

    # Credit Section
    with credit_column:
        st.write("*Created with :heart: by [Moyo](https://github.com/Moyo-tech)*")   
        
    st.write(
        "Welcome to DermDetect AI!:hospital: This application uses a deep learning model to classify skin cancer images into different categories. Upload an image or select an existing image to test, and the model will provide predictions along with confidence levels."
    )
    # Define the mapping of short-form labels to long-form labels
    label_mapping = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }
     # Initialize session state
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = None
        
    with st.spinner('Application Loading...'):
        # Load the model using st.cache
        @st.cache(allow_output_mutation=True)
        def load_model1():
            custom_objects = {
                'top_2_accuracy': top_2_accuracy,
                'top_3_accuracy': top_3_accuracy,
            }
            return load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        
    model = load_model1()
    categories = load_labels(LABELS_PATH)

    choice = st.radio("Choose an option", ["Upload Image", "Select Existing Image"])
    
    if choice == "Upload Image":
        patient_info = gather_patient_info()
        image = load_image()
        if image is not None:
            
            st.session_state.patient_info = patient_info
            result = st.button('Run on image')
            if result:
                predictions = predict(model, label_mapping, categories, image, patient_info, selected_choice=choice)
    else:
        image = load_existing()
        if image is not None:
            result = st.button('Run on image')
            if result:
                predict(model, label_mapping, categories, image, selected_choice=choice)
    st.markdown("***")
    st.markdown(
        "Thanks for testing this out! I'd love feedback on this, so if you want to reach out you can find me on [LinkedIn](https://www.linkedin.com/in/moyosoreweke/) or check out my other projects on [Github](https://github.com/Moyo-tech)."
    )
            
    gc.collect()

if __name__ == '__main__':
    main()
