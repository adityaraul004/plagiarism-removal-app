import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import torch

# Ensure required nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

# Load the paraphrasing model
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
paraphrase_model = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", device=device)

def plagiarism_removal(text):
    """Paraphrases text using a transformer-based model."""
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    if not filtered_text.strip():
        return "No valid words to process."
    
    try:
        result = paraphrase_model(filtered_text, max_length=100, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error in processing: {e}"

# Streamlit UI
st.title("📝 AI-Powered Plagiarism Remover")
st.markdown("Enter your text below and get a rephrased version to reduce plagiarism.")

# User Input
input_text = st.text_area("Enter text to paraphrase:", height=200)

if st.button("Paraphrase Text"):
    if input_text.strip():
        modified_text = plagiarism_removal(input_text)
        st.subheader("Rephrased Text:")
        st.write(modified_text)
        st.button("Copy to Clipboard", on_click=lambda: st.write("Copied!"))
    else:
        st.warning("⚠️ Please enter some text before clicking the button.")

st.markdown("🚀 Powered by **Hugging Face Transformers** & **Streamlit**")

