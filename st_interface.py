import warnings
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from utils import split_character, do_predict, load_model, example2file

app_title = "A Robust License Plate Recognition System based on Domain Adaptation"


@st.cache(show_spinner=False)
def load_local_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = np.array(Image.open(BytesIO(bytes_data)).convert("RGB"))
        return image
    else:
        return None


def main():
    st.set_page_config(page_title=app_title)
    image = None  # init

    # uploaded_file = st.sidebar.file_uploader(" ")

    model_name = st.sidebar.selectbox(
        "Select a recognition model",
        ("Logistic Regression", "SVM", "Domain-Adversarial Neural Networks", "Correlation Alignment (CORAL)"),
    )
    use_example = st.sidebar.selectbox("Choose an image to recognize", ["Upload by myself"] + [i for i in example2file])

    if use_example == "Upload by myself":
        uploaded_file = st.file_uploader(
            "Upload a licence plate image to do the recognition (current trained model for Chinese licence plate):",
            type=["png", "jpg"],
        )
        image = load_local_image(uploaded_file)
    else:
        image = Image.open("examples/" + example2file[use_example]).convert("RGB")

    if image is not None:
        st.image(image, caption="Input license plate image")
        # st.write("Debug: image shape:", image.shape)

    if st.button("Recognize") and image is not None:
        cropped_list = split_character(image)
        if not len(cropped_list):
            st.write("Sorry, splitting process failed, please try another image.")
        else:
            model = load_model(model_name)
            if model is None:
                show_str = "Model is still under construction, please try another one now."
            else:
                results = do_predict(model, cropped_list)
                show_str = "The recognized result is: {"
                for each in results:
                    show_str += each + " "
                show_str += "}"
            st.write(show_str)


if __name__ == "__main__":
    main()
