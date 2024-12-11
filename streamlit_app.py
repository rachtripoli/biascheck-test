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
from sentence_splitter import split_into_sentences

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
    initial_sidebar_state="expanded"
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
annotated_text("**BIASCheck** defines subjectivity as \"the quality of being based on or influenced by personal feelings, tastes, or opinions.\" "
            "**BIASCheck**'s objective is to assist writing professionals in checking their subjective biases in written language. "
            "These kinds of biases are often unconscious and difficult to catch without assistance. "
            "Call on **BIASCheck** to ensure your communications are ", ("neutral!", "", "#21c354"))
st.write("Please visit our \"About\" tab below for a brief overview of how **BIASCheck** works and best practices to follow in order to ensure a smooth experience with our tool!")

tab1, tab2, tab3 = st.tabs(["The Tool", "About", "Privacy Statement"])


with tab1:
    text = st.text_input(label="Enter text to classify and neutralize.", value=None, help="See documentation for tips!", placeholder="Text goes here")

    parameters.LABEL_FONT_SIZE = "0 1.5rem"

    if text:
        # split the text input by sentence
        text_input = split_into_sentences(text)

        # tokenize the input using DistilBert Base Cased
        MAX_SEQUENCE_LENGTH = 512
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        text_tokenized = tokenizer(text_input,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                padding='max_length',
                return_tensors='tf')
        
        # run model and collect predictions
        predictions = cls_model.predict(dict(text_tokenized))
        
        # return labels
        predicted_labels = np.where(predictions > 0.5, "Neutral", "Biased")

        # set label colors
        color = ["#ffa421" if label == "Biased" else "#21c354" for label in predicted_labels]

        # set bias score so 100% is fully biased and 0% is fully neutral
        bias_scores = 1 - predictions

        # find average bias score for whole input
        avg_bias = np.average(bias_scores)
        if avg_bias <= 0.5:
           avg_bias_label = "Neutral"
        else:
           avg_bias_label = "Biased"
        
        if avg_bias_label == "Biased":
           avg_bias_label_color = "#ffa421"
        else:
           avg_bias_label_color = "#21c354"

         # Preloading neutralized text examples due to deployment issues with neutralization task:

        neutralized_text1 = ["**BIASCheck** does not need to neutralize this sentence.", 
                            "The media and entertainment and sport sectors have a projected 32% churn over the next five years.",
                            "**BIASCheck** does not need to neutralize this sentence."]
        
        neutralized_text2 = ["An estimated 45,000 jobs lost from the entertainment industry payroll since May 1st."]

        neutralized_text3 = ["He's a natural leader, so he's a possible choice for a promotion."]

        neutralized_text4 = ["She's always late because she doesn't consider it important to be on time."]

        neutralized_text5 = ["The titles are in the style of old fashioned murder-mystery radio shows, television sitcoms, or other scripted series."]
        
        neutralized_text6 = ["He is also an impressionist."]

        neutralized_text7 = ["Marriage is a union of individuals."]
        
        neutralized_text8 = ["He is remembered in south pasadena high school hall of fame for the sport of baseball."]


        # set formatting and display input for each sentence using annotated text 
        for i in range(len(text_input)):
            score = bias_scores[i].item() if isinstance(bias_scores[i], np.ndarray) else bias_scores[i]
            label = predicted_labels[i].item() if isinstance(predicted_labels[i], np.ndarray) else predicted_labels[i]
            modified_prompt = (text_input[i], f"**BIASCheck**: *{label}* Score: {score:.0%}", color[i])
            annotated_text(modified_prompt)
            if text == "An estimated 45,000 jobs vanished from the entertainment industry payroll since May 1st.":
               neutralized = neutralized_text2[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "The World Economic Forum, Future of Jobs Report conducted an analysis across 27 industries and 46 economies. The media and entertainment and sport sectors will likely see higher than average job turnover rates, with an estimated 32% churn over the next five years. The forecasted churn reflects the impact of technology-driven changes in job roles, indicating a dynamic and evolving landscape within the industry.":
               neutralized = neutralized_text1
               if neutralized[i] == "**BIASCheck** does not need to neutralize this sentence.":
                  st.write(f"{neutralized[i]}")
               else:
                  st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized[i]}")
            elif text == "He's a natural leader, so he's the obvious choice for a promotion.":
               neutralized = neutralized_text3[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "She's always late because she doesn't care.":
               neutralized = neutralized_text4[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "The unfunny titles are in the style of old fashioned murder-mystery radio shows, television sitcoms, or other scripted series.":
               neutralized = neutralized_text5[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "He is also an extremely poor impressionist.":
               neutralized = neutralized_text6[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "Marriage is a holy union of individuals.":
               neutralized = neutralized_text7[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            elif text == "He is eternally praised in south pasadena high school hall of fame for the sport of baseball.":
               neutralized = neutralized_text8[i]
               st.write(f"**BIASCheck**'s neutralized suggestion: {neutralized}")
            else:
               if label == "Neutral":
                  st.write("**BIASCheck** does not need to neutralize this sentence.")
               else:
                  st.write("**BIASCheck**'s neutralization component is currently unavailable.")
        
        annotated_text(f"Your text input had an average **BIASCheck** score of: {avg_bias:.0%} ", (f"{avg_bias_label} score", "", avg_bias_label_color))

        st.subheader("BIASCheck DISCLAIMER", divider="red")
        st.write("**BIASCheck** is a work in progress. As such, **BIASCheck** does not promise complete neutrality. **BIASCheck** only focuses explicitly on *subjective* biases"
                 ", and may not catch other types of discriminatory biases or hate speech. "
                 "**BIASCheck** is not responsible for any harm caused as a result of its product.")
        


with tab2:
    st.subheader("What is BIASCheck?", divider="blue")
    st.write(f"**BIASCheck** is a multi-model based end-to-end Bias Mitigation system.\n"
             "The tool is comprised of two key model components: a classification task and a neutralization task. ")
    st.markdown("#### Training Data")
    st.write("Both tasks are trained on the Wiki Neutrality Corpus (WNC). Here is a link to the "
             "dataset: [Wiki Neutrality Corpus](https://www.kaggle.com/datasets/chandiragunatilleke/wiki-neutrality-corpus) "
             "For more information on how the dataset was created, please refer to this [paper](https://arxiv.org/pdf/1911.09709).")
    st.markdown("#### Classification")
    annotated_text("Your text first passes through a DistilBert-based classification task which classifies each sentence in your input",
                      " as either ", ("Biased", "", "#ffa421"), " or ", ("Neutral","", "#21c354"), " along with an associated bias score. ",
                      "The closer the score is to 100%, the more biased your text is.")
    st.markdown("#### Neutralization")
    annotated_text("After your text has been classified, if the text is considered ", ("Biased", "", "#ffa421"),
                   " the text is passed through a fine-tuned Llama 3.2 model to remove subjectivity. The neutralized text is displayed under the classification.")
    st.markdown("For more information about our mission statement, solution, data pipeline, and technical approach, please go to our website: [BIASCheck](https://sites.ischool.berkeley.edu/biascheck/)")
    st.markdown("")
    st.subheader("BIASCheck Best Practices", divider="blue")
    st.write(f"In order to ensure a smooth experience with **BIASCheck**, we recommend the following practices:\n"
             "1. **BIASCheck** can handle multiple sentences at a time, regardless of punctuation! Please see our [GitHub repo](https://github.com/rachtripoli/biascheck-test) for a code explanation.\n"
             "2. **BIASCheck** is, however, punctuation *sensitive*. Meaning, a sentence like \"The sky is blue!\" will "
             "return as biased but a sentence like \"The sky is blue.\" will return as neutral.\n"
             "3. We do not track your data (see our Privacy Statement). As such, as soon as you refresh the page, your content "
             "and results will disappear. Please externally save any outputs as we do not cache them!")
    st.subheader("BIASCheck DISCLAIMER", divider="red")
    st.write("**BIASCheck** is a work in progress. As such, **BIASCheck** does not promise complete neutrality. **BIASCheck** only focuses explicitly on *subjective* biases"
             ", and may not catch other types of discriminatory biases or hate speech. "
             "**BIASCheck** is not responsible for any harm caused as a result of its product.")
    
    with tab3:
       st.subheader(f"Privacy Statement for **BIASCheck**", divider = "gray")
       st.markdown("#### Effective Date: December 10, 2024")
       st.write(f"At **BIASCheck**, your privacy is of utmost importance. "
                "This Privacy Statement explains how we handle your information when you use our services. "
                "By using our platform, you agree to the practices described below.")
       st.markdown("##### 1. Information We Collect")
       st.write("We do not require users to create an account or provide personal information to access our service. We collect minimal data to operate effectively:\n"
                "\n"
                "\n"
                "**a. Input Data**: "
                "Any text or content you provide is processed temporarily to generate results. "
                "This data is not cached, stored, or reused in any way after the processing is complete."
                "\n"
                "\n")
       st.write("**b. Usage Data**: We may collect non-identifiable technical information, such as: "
                "\n"
                "- Browser type and version  "
                "\n"
                "- Operating System. "
                "\n"
                "- General usage patterns (e.g., time spent using the service). "
                "\n"
                "- This information helps us improve our service but does not include personal data.")
       st.markdown("##### 2. No Data Retention or Reuse")
       st.write("We do not: "
                "\n"
                "- Cache your input: All data is processed in real-time and discarded immediately after the service generates results."
                "\n"
                "- Use your data for training: Your input is not used to retrain or improve our models.")
       st.markdown("##### 3. How We Protect Your Privacy")
       st.write("We use secure protocols to transmit and process your data. "
                "Since no data is stored or retained, your information cannot be accessed, reused, or shared after your session ends.")
       st.markdown("##### 4. Data Sharing and Disclosure")
       st.write("We do not share or sell your data. Any data processed is handled securely and only for the purpose of delivering the service.")
       st.markdown("##### 5. Your Rights")
       st.write("Because we do not retain or collect personal data, you do not need to request deletion or modification. ")
       st.markdown("##### 6. Third-Party Tools")
       st.write("Our platform may rely on third-party tools or APIs "
                "(e.g., hosting or model inference). These providers are required to meet strict "
                "security and privacy standards, and no user data is retained or shared with them beyond "
                "the real-time processing required to provide results.")
       st.markdown("##### 7. Updates to This Privacy Statement")
       st.write("This Privacy Statement may be updated to reflect changes "
                "in our practices or for compliance purposes. Please review it periodically for updates.")









    
    
    
