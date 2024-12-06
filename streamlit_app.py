import streamlit as st
from annotated_text import annotated_text, parameters 
import numpy as np
import pandas as pd

data={"example_id": [0, 1], "original_text":["Creation science (or cs) is an unscientific effort to provide evidence supporting the account of the creation of the universe related in the bible.", "Hyman Bloom (b. Brunavii, Latvia, March 29, 1913) is a painter."], "neutralized_text":["Creation science (or cs) is an effort to provide evidence supporting the account of the creation of the universe related in the bible.", "Neutral"], "bias_detection": ["Biased", "Neutral"], "bias_score": [0.67, 1-0.997]}

df=pd.DataFrame(data=data)

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

text = st.selectbox(label="Select a text example to detect and neutralize.", options=df["original_text"], index=None)

parameters.LABEL_FONT_SIZE = "0 1.5rem"

if text:
    prompt_split = text
    bias_detection = df.loc[df['original_text'] == text, 'bias_detection'].iloc[0]
    if bias_detection == "Biased":
        color = "#ffa421"
    else:
        color = "#21c354"
    bias_score = df.loc[df['original_text'] == text, 'bias_score'].iloc[0]
    modified_prompt = (prompt_split, f"Label: {bias_detection}. Score: {bias_score: .1%}", color)
    annotated_text(modified_prompt)
    if bias_detection == "Biased":
        neutral = df.loc[df['original_text'] == text, 'neutralized_text'].iloc[0]
        st.markdown("**BIASCheck's neutralized version is**: \n{}".format(neutral))
    else:
        st.markdown("**BIASCheck has determined this text is neutral.**".format())