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


def is_relevant_image(test_image):
    base_image = cv2.imread('models/Image1.jpg')

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

def predict(model, label_mapping, categories, image):
    img_array = preprocess_image(image)
    
    with st.spinner('Classifying...'):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

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

def main():
    gc.enable()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    st.set_page_config(page_title='DermDetect', page_icon=':microscope:')

    st.title('DermDetect AI :microscope:')
    st.write(
        "Welcome to DermDetect AI!:hospital: This application uses a deep learning model to classify skin cancer images into different categories. Upload an image or select an existing image, and the model will provide predictions along with confidence levels."
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
        image = load_image()
        if image is not None:
            result = st.button('Run on image')
            if result:
                predict(model, label_mapping, categories, image)
    else:
        image = load_existing()
        if image is not None:
            result = st.button('Run on image')
            if result:
                predict(model, label_mapping, categories, image)

    gc.collect()

if __name__ == '__main__':
    main()
