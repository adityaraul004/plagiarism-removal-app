import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import pipeline  # Hugging Face's paraphraser
import torch

# Ensure required nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

# Load paraphrasing model
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

def process_text():
    """Processes user input and updates the output text box."""
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Error", "Please enter text before processing.")
        return
    
    modified_text = plagiarism_removal(input_text)
    
    text_output.config(state=tk.NORMAL)
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, modified_text)
    text_output.config(state=tk.DISABLED)

def copy_to_clipboard():
    """Copies the output text to clipboard."""
    output_text = text_output.get("1.0", tk.END).strip()
    if output_text:
        root.clipboard_clear()
        root.clipboard_append(output_text)
        root.update()
        messagebox.showinfo("Success", "Text copied to clipboard!")

# GUI Setup
root = tk.Tk()
root.geometry('800x500')
root.title('Plagiarism Removal Tool')

# Configuring grid layout for responsiveness
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(3, weight=1)

tk.Label(root, text="Enter text to remove plagiarism:", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=5)
text_input = scrolledtext.ScrolledText(root, height=6, wrap=tk.WORD)
text_input.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

process_button = tk.Button(root, text="Remove Plagiarism", command=process_text, font=("Arial", 12), bg="lightblue")
process_button.grid(row=2, column=0, columnspan=2, pady=5)

tk.Label(root, text="Modified Text:", font=("Arial", 12)).grid(row=3, column=0, columnspan=2, pady=5)
text_output = scrolledtext.ScrolledText(root, height=6, wrap=tk.WORD, state=tk.DISABLED)
text_output.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard, font=("Arial", 12), bg="lightgreen")
copy_button.grid(row=5, column=0, columnspan=2, pady=5)

root.mainloop()
