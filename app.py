import streamlit as st
import os
import pickle
import requests

# Model download settings
MODEL_URL = "https://www.dropbox.com/scl/fi/ufmclyehxqk1vrgxkomci/Fund_Allocation.pk1?rlkey=63rr8tjq7kq7fr6sowhi42fvu&st=tb34kq99&dl=1"
MODEL_PATH = "Fund_Allocation.pk1"

@st.cache_resource
def load_model():
    """Load the ML model from the local file."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... This may take a few minutes.")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download the model. Status code: {response.status_code}")
            st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()
st.success("Model loaded successfully!")
