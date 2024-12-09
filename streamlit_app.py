import streamlit as st
from annotated_text import annotated_text, parameters 
import numpy as np
import pandas as pd
import os
from transformers import DistilBertTokenizer
from tensorflow import keras
from keras import models
import transformers
from st_files_connection import FilesConnection
import tempfile
import zipfile
import s3fs

# set aws credentials
aws_access_key_id=st.secrets["aws_access_key_id"]
aws_secret_access_key=st.secrets["aws_secret_access_key"]
region_name="us-east-1"
BUCKET_NAME = "biascheck-232442840523-us-east-1"
model_name = "classification_model_12082024.zip"

# define s3 bucket functions
def get_s3fs():
  return s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

def s3_get_keras_model(model_name: str) -> keras.Model:
  with tempfile.TemporaryDirectory() as tempdir:
    s3fs = get_s3fs()
    # Fetch and save the zip file to the temporary directory
    s3fs.get(f"s3://{BUCKET_NAME}/{model_name}", f"{tempdir}/{model_name}.zip")
    # Extract the model zip file within the temporary directory
    with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir}/{model_name}")
    # Load the keras model from the temporary directory
    return keras.models.load_model(f"{tempdir}/{model_name}", custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})

# load model using previous function  
cls_model=s3_get_keras_model(model_name)

# load biascheck logo
from PIL import Image
img = Image.open("biaschecklogo.png").convert('RGBA')

# set page configuration
st.set_page_config(
    page_title="BIASCheck",
    page_icon=img,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

col1, col2 = st.columns([0.09, 0.91], vertical_alignment="center")

scale_factor = 0.1
new_width = int(img.width * scale_factor)
new_height = int(img.height * scale_factor)
img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

with col1:
    st.image(img_resized)
with col2:
    st.title("BIASCheck")
    
st.subheader("Minimizing subjectivity in language.")
st.markdown("BIASCheck's objective is to assist writing professionals in checking their subjective biases in written language. Subjective biases are often unconscious and difficult to catch without a second pair of eyes. In the times when you are unable to call on someone else to read your work, call on BIASCheck!")

tab1, tab2, tab3 = st.tabs(["The Tool", "About", "Privacy Statement"])


with tab1:
    text = st.text_input(label="Enter a text example to classify.", value=None, help="See documentation for tips!", placeholder="Text goes here")

    parameters.LABEL_FONT_SIZE = "0 1.5rem"

    if text:
        text_input = text.split(". ")
        MAX_SEQUENCE_LENGTH = 512
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        text_tokenized = tokenizer(text_input,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='tf')
        predictions = cls_model.predict(dict(text_tokenized))
        
        # predictions_flat = predictions.flatten()
        # predictions_list = predictions_flat.tolist()
        predicted_labels = np.where(predictions > 0.5, "Neutral", "Biased")
        # labels_flattened = predicted_labels.flatten().tolist()
        # if predicted_labels == "Biased":
        #     color = "#ffa421"
        # else:
        #     color = "#21c354"
        #[f(x) if condition else g(x) for x in sequence]

        color = ["#ffa421" if label == "Biased" else "#21c354" for label in predicted_labels]
        bias_scores = 1 - predictions

        for i in range(len(text_input)):
            score = bias_scores[i].item() if isinstance(bias_scores[i], np.ndarray) else bias_scores[i]
            label = predicted_labels[i].item() if isinstance(predicted_labels[i], np.ndarray) else predicted_labels[i]
            modified_prompt = (text_input[i], f"**BIASCheck**: *{label}* Score: {score:.0%}", color[i])
            annotated_text(modified_prompt)
        # modified_prompt = (text, f"**BIASCheck**: *{labels_flattened[0]}* Score: {bias_score:.0%}", color)
        # annotated_text(modified_prompt)

with tab2:
    st.subheader("What is BIASCheck?", divider="blue")
    st.write(f"**BIASCheck** is a multi-model based end-to-end Bias Mitigation system.\n"
             "The tool is comprised of two key model components: a classification task and a neutralization task. ")
    st.markdown("#### The Classification Task")
    annotated_text("Your text first passes through a DistilBert-based classification task which classifies each sentence in your input",
                      " as either ", ("Biased", "", "#ffa421"), " or ", ("Neutral","", "#21c354"), " along with an associated bias score. ",
                      "The closer the score is to 100%, the more biased your text is.")
    st.markdown("#### The Neutralization Task")
    annotated_text("After your text has been classified, if the text is considered ", ("Biased", "", "#ffa421"),
                   " the text is passed through a fine-tuned Llama model to remove subjectivity. The neutralized text is displayed under the classification.")
    st.markdown("For more information on our data pipeline, technical approach, and decision making, please go to our website: [BIASCheck](https://sites.ischool.berkeley.edu/biascheck/)")
    st.markdown("")
    st.subheader("BIASCheck Best Practices", divider="blue")
    st.write(f"In order to ensure a smooth experience with **BIASCheck**, we recommend the following practices:\n"
             "1. **BIASCheck** can handle multiple sentences at a time, but all sentences must end in a period in order for the tool to properly detect.\n"
             "2. ")
    
    with tab3:
       st.subheader("Privacy Statement", divider = "gray")
       st.markdown("#### Privacy Statement")
    
    
    
