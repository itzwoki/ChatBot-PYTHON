import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

import tkinter as tk
from tkinter import font
from tkinter import ttk

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# GUI Initialization
window = tk.Tk()
window.title("HAWK Bot")
window.geometry("500x700")
window.configure(bg="#292929")

# Chat Display
chat_frame = tk.Frame(window, bg="#333333")
chat_frame.pack(pady=10, padx=10)

response_text = tk.Text(chat_frame, bg="#333333", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0, state=tk.DISABLED)
response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
response_text.tag_configure('user_message', foreground="green")
response_text.tag_configure('bot_message', foreground="white")

# Scrollbar
scrollbar = ttk.Scrollbar(chat_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
response_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=response_text.yview)

def display_user_message(message):
    response_text.configure(state='normal')
    response_text.insert(tk.END, "\nYou: " + message + '\n', 'user_message')
    response_text.configure(state='disabled')
    response_text.see(tk.END)

def display_bot_message(message):
    response_text.configure(state='normal')
    response_text.insert(tk.END, "\nHAWK: " + message + '\n', 'bot_message')
    response_text.configure(state='disabled')
    response_text.see(tk.END)

def send_message(event=None):
    message = entry.get().strip()
    if message:
        display_user_message(message)
        entry.delete(0, tk.END)
        display_bot_message("Bot is typing...")
        window.after(1000, generate_response, message)

def generate_response(message):
    response = get_response(predict_class(message), intents)
    response_text.configure(state='normal')
    response_text.delete("end-2l", tk.END)  # Delete "Bot is typing..."
    response_text.insert(tk.END, "HAWK: " + response + '\n', 'bot_message')
    response_text.configure(state='disabled')
    response_text.see(tk.END)

# User Input
input_frame = tk.Frame(window, bg="#333333")
input_frame.pack(pady=10, padx=10)

entry = tk.Entry(input_frame, font=("Helvetica", 12), fg="white", bg="#555555", relief=tk.FLAT)
entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def on_send_enter(event):
    send_message()

entry.bind("<Return>", on_send_enter)

def on_send_hover(event):
    send_button.config(bg="#444444")

def on_send_leave(event):
    send_button.config(bg="#555555")

send_button = tk.Button(input_frame, text="Send", font=("Helvetica", 12), bg="#555555", fg="white", relief=tk.FLAT, command=send_message)
send_button.pack(side=tk.LEFT, padx=(5, 0))
send_button.bind("<Enter>", on_send_hover)
send_button.bind("<Leave>", on_send_leave)

# Hover Effects for Send Button
def on_send_hover(event):
    send_button.config(bg="#444444")

def on_send_leave(event):
    send_button.config(bg="#555555")

send_button.bind("<Enter>", on_send_hover)
send_button.bind("<Leave>", on_send_leave)

# Run the GUI
window.mainloop()
