import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the pre-trained model
model = load_model('iimc.h5')

# Define the class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title('Intel Image Classification')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    # Display the prediction
    st.write(f"Prediction: {class_names[predicted_class]}")

    # Display the confidence of each class
    st.write("Class probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]:.4f}")

# Display confusion matrix
if st.button('Show Confusion Matrix'):
    test_dir = 'iimcd\seg_test\seg_test'

    test_datagen = image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)

    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt)
