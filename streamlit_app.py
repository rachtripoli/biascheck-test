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

# aws_access_key_id = st.secrets["aws_access_key_id"]
# aws_secret_access_key = st.secrets["aws_secret_access_key"]
# region = st.secrets["region"]

# session = boto3.Session(profile_name="streamlit1")
# s3 = session.client("s3",
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=region)

# url = s3.generate_presigned_url(
#     ClientMethod="get_object",
#     Params={"Bucket": "biascheck-232442840523-us-east-1", "Key": "distilbert_cls_model.h5"},
#     ExpiresIn=3600
# )

# # Download and Load Model
# response = requests.get(url)
# with open("distilbert_cls_model.h5", "wb") as f:
#     f.write(response.content)

conn = st.connection('s3', type=FilesConnection)
model1 = conn.open("biascheck-232442840523-us-east-1/distilbert_cls_model.h5")

cls_model = models.load_model(model1, custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})

from PIL import Image
img = Image.open("biaschecklogo.png").convert('RGBA')

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

scale_factor = 2.5
new_width = int(img.width * scale_factor)
new_height = int(img.height * scale_factor)
img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

with col1:
    st.image(img_resized)
with col2:
    st.title("BIASCheck")
    
st.subheader("Minimizing subjectivity in language.")
st.markdown("BIASCheck's objective is to assist writing professionals in checking their subjective biases in written language. Subjective biases are often unconscious and difficult to catch without a second pair of eyes. In the times when you are unable to call on someone else to read your work, call on BIASCheck!")

text = st.text_input(label="Enter a text example to classify.", value=None, help="See documentation for tips!", placeholder="Text goes here")

parameters.LABEL_FONT_SIZE = "0 1.5rem"

if text:
    MAX_SEQUENCE_LENGTH = 512
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    text_tokenized = tokenizer(text,
              max_length=MAX_SEQUENCE_LENGTH,
              truncation=True,
              padding='max_length',
              return_tensors='tf')
    predictions = cls_model.predict(dict(text_tokenized))
    predictions_flat = predictions.flatten()
    predictions_list = predictions_flat.tolist()
    predicted_labels = np.where(predictions > 0.5, "Neutral", "Biased")
    labels_flattened = predicted_labels.flatten().tolist()
    if predicted_labels == "Biased":
        color = "#ffa421"
    else:
        color = "#21c354"
    bias_score = 1 - predictions_list[0]
    modified_prompt = (text, f"**BIASCheck**: *{labels_flattened[0]}* Score: {bias_score:.0%}", color)
    annotated_text(modified_prompt)