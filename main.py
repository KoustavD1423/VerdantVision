import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('M:\\Projects\\machineLearning\\Plant Village Detection Using CNn\\video\\CNN_Esca_models\\CNN_Esca_model.keras')  

# Define class names
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img, img_array

# Function to make prediction and display the result
def predict_and_display(img_array):
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)
    model_prediction = class_name[result_index]
    return model_prediction

# Custom CSS for background and other styles
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #2b5876, #4e4376);
            color: white;
        }
        .sidebar .sidebar-content {
            background: #1e1e2f;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .css-18e3th9 {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }
        .css-1d391kg {
            color: white;
        }
        .css-1d391kg:hover {
            background: #4e4376;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # Make sure this image exists in your directory
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
        This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purpose.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("### Upload an image of a plant leaf to predict its health status:")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, width=400, use_column_width=True)
        
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(test_image.getbuffer())
        
        img, img_array = load_and_preprocess_image("temp_image.jpg")
        
        # Predict button
        if st.button("Predict"):
            with st.spinner('Analyzing the image...'):
                model_prediction = predict_and_display(img_array)
            st.success(f"Model prediction: **{model_prediction.replace('_', ' ')}**")
            st.balloons()
            st.markdown(f"""
                ### Recommendation:
                - Ensure proper care of the plant to prevent the spread of {model_prediction.split('___')[1].replace('_', ' ')}.
                - Regularly monitor and apply suitable treatments as advised by agricultural experts.
            """)
