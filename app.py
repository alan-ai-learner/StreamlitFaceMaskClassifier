import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
##
st.sidebar.header("About")
st.sidebar.text("Streamlit App \n Made by Alankar Shukla")
st.sidebar.markdown('[Visit my Github Account and give Star.](github.com/alan-ai-learner/StreamlitFaceMskClassifier)')

html_temp = """
    <div style="background-color:tomato;padding:15px;">
    <h1>Face Mask Classifier</h1>
    </div)  
    """
#

def prediction(data):
    
    model = load_model('final_model_vggNet19.h5')
    test_image = data.resize((224,224), Image.ANTIALIAS)
    test_image = test_image.convert("RGB")
    test_image = image.img_to_array(test_image)    
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    values = ['with_mask','without_mask']
    result = { "image" : values[result.argmax()]}
    return result

def upload():
    uploaded_file = st.file_uploader("   ", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        return img


def main():
    """Run this function to display the Streamlit app"""
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("   ", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)     
        st.image(img, caption='Uploaded Image.')    
    if st.button('Click to Classify!'):
        if prediction(img)['image'] == 'with_mask':
            st.success('Good! you are wearing a mask.')
            st.balloons()
        else:
            st.warning('You are not wearing a mask! Please wear a mask.')


if __name__ == "__main__" :
    main()