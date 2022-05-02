import warnings
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

app_title = "A Robust License Plate Recognition System based on Domain Adaptation"


@st.cache(show_spinner=False)
def load_local_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()  
        image = np.array(Image.open(BytesIO(bytes_data)))
        return image
    else:
        return None

def main():
    st.set_page_config(page_title=app_title)

    uploaded_file = st.file_uploader("Upload a licence plate picture to do the recognition", type=['png', 'jpg'])
    uploaded_file = st.sidebar.file_uploader(" ")
    image = load_local_image(uploaded_file)
    
    if st.button('Recognize') and image is not None:
        st.write('Results')

    if image is not None:
        st.image(image)

if __name__ == "__main__":
    main()