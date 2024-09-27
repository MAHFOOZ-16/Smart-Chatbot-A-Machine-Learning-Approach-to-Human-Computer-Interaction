import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import colorama
from colorama import Fore, Style
import random

# Initialize colorama for terminal colors
colorama.init()

## Read the intents JSON file
try:
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'intents.json' file not found.")
    exit()

# Initialize lists for training
training_sentences = []
training_labels = []
labels = []
responses = []

# Extracting the patterns and responses from intents
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Number of unique labels (classes)
num_classes = len(labels)

## Using LabelEncoder to encode the target labels
lbl_encoder = LabelEncoder()
training_labels = lbl_encoder.fit_transform(training_labels)

# Tokenization (Vectorize the text data)
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"  # Out of Vocabulary token

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    LSTM(64),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Training the model
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# Save the trained model
model.save("chat_model.keras")

# Save the tokenizer object and label encoder
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


## Chat function to interact with the user
def chat():
    # Load the trained model
    model = keras.models.load_model('chat_model.keras')

    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoder
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # Parameters
    max_len = 20

    # Start the chat
    print(Fore.YELLOW + "Start messaging with the bot (type 'quit' to stop)!" + Style.RESET_ALL)
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()

        # Exit condition
        if inp.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break

        if inp.strip() == "":
            print(Fore.RED + "ChatBot: Please type something!" + Style.RESET_ALL)
            continue
        
        sequence = tokenizer.texts_to_sequences([inp])
        padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)

        result = model.predict(padded_sequence)
   
        # Preprocess input and make prediction
        result = model.predict(keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))

        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # Respond with a random response for the detected intent
        for intent in data['intents']:
            if intent['tag'] == tag[0]:
                print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL, random.choice(intent['responses']))


# Start the chat session
chat()
