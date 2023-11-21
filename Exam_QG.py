


import spacy
import textract
import PyPDF2
import random

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext

# Set a random seed
random.seed(8)

from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

def generate_quiz_questions(input_type, input_path, limit=None):
    if input_type == 'txt':
        notes_text = read_text_file(input_path)
    elif input_type == 'pdf':
        notes_text = extract_text_from_pdf(input_path)
    else:
        raise ValueError("Invalid input type. Supported types: 'text', 'pdf'")

    doc = nlp(notes_text)

    # Create a list of sentences for TF-IDF analysis
    sentences = [sent.text for sent in doc.sents]

    # Calculate TF-IDF scores for each term
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    feature_names = vectorizer.get_feature_names_out()

    # Calculate the sum of TF-IDF scores for each term
    term_scores = tfidf_matrix.sum(axis=0).tolist()[0]

    term_tfidf = {feature_names[i]: term_scores[i] for i in range(len(feature_names))}

    topics = list(set(token.text for token in doc if token.pos_ == "NOUN" and token.text in term_tfidf))
    topics = [topic for topic in topics if term_tfidf.get(topic, 0) > 0.2]  # Adjust the threshold as needed

    if limit is not None and limit < len(topics):
        topics = topics[:limit]
    questions = [f"What is {topic}?" for topic in topics]
    
    random.shuffle(questions)

    return questions


# Interface
class QuestionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quiz Question Generator")

        self.label = tk.Label(root, text="Welcome to Quiz Question Generator!")
        self.label.pack(pady=10)

        self.file_button = tk.Button(root, text="Open File", command=self.open_file_dialog)
        self.file_button.pack(pady=10)

        self.result_text = tk.Text(root, height=5, width=40, state='disabled')
        self.result_text.pack(pady=10)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(title="Select a File")

        self.generate_questions(file_path[-3:], file_path, limit=5)
        messagebox.showinfo("Success", "Questions generated successfully!")

    def generate_questions(self, input_type, input_path, limit=5):
        questions = generate_quiz_questions(input_type, input_path, limit)
        self.display_results("Generated Questions:\n" + "\n".join(questions))

    def display_results(self, message):
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, message)
        self.result_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = QuestionGeneratorApp(root)
    root.mainloop()