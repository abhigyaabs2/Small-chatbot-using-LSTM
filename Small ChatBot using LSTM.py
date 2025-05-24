#!/usr/bin/env python
# coding: utf-8

# In[1]:


#User Input ➜ Tokenizer ➜ Encoder (LSTM) ➜ Context Vector ➜ Decoder (LSTM) ➜ Response


# In[1]:


import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


# Step 2: Load and Clean Dialog Data

questions = [
    "Hi!",
    "How are you?",
    "What are you doing?",
    "Who are you?",
    "Tell me a joke",
    "Bye"
]

answers = [
    "Hello!",
    "I am fine, Thank you!",
    "Just thinking about you.",
    "I am batman.",
    "Why did the chicken cross the road? To get to KFC!",
    "Goodbye!"
]


# In[3]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?!.]+", " ", text)
    return text

questions = [clean_text(q) for q in questions]
answers = ["<START> " + clean_text(a) + " <END>" for a in answers]


# In[4]:


# Step 3: Tokenize and Pad Sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Convert to sequences
questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)

#Padding
max_len = 10
encoder_input = pad_sequences(questions_seq, maxlen=max_len, padding='post')
decoder_input = pad_sequences([seq[:-1] for seq in answers_seq], maxlen=max_len, padding='post')
decoder_output = pad_sequences([seq[1:] for seq in answers_seq], maxlen=max_len, padding='post')


# In[5]:


# Step 4: Build the Seq2Seq Model

embedding_dim = 64
lstm_units = 128

#Encoder
encoder_inputs = Input(shape=(max_len,))
enc_emb = Embedding(VOCAB_SIZE, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c, = LSTM(lstm_units, return_state=True)(enc_emb)

#Decoder
decoder_inputs = Input(shape=(max_len,))
dec_emb = Embedding(VOCAB_SIZE, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True)
decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
output_layer = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = output_layer(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()


# In[6]:


# Step 5: Train the Model
decoder_output = np.expand_dims(decoder_output, -1)
model.fit([encoder_input, decoder_input], decoder_output, epochs=300, verbose=0)
print("Training complete")


# In[7]:


# Step 6: Build a Simple Chatbot Response Generator
def decode_sequence(input_seq):
    states_value = model.layers[2].output[1:] 
    input_seq = pad_sequences(tokenizer.texts_to_sequences([clean_text(input_seq)]), maxlen=max_len, padding='post')
    
    # Start token
    target_seq = np.zeros((1, max_len))
    target_seq[0, 0] = tokenizer.word_index['start']
    
    stop_condition = False
    decoded = ''
    for i in range(1, max_len):
        output_tokens = model.predict([input_seq, target_seq])[0]
        sampled_token_index = np.argmax(output_tokens[0, i-1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'end' or sampled_word == '':
            break
        decoded += sampled_word + ' '
        target_seq[0, i] = sampled_token_index

    return decoded.strip()


# In[8]:


while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("Bot: Goodbye!")
        break
    elif user_input.lower() in ['hi', 'hello']:
        print("Bot: Hello!")
        break
    elif user_input.lower() in ['who are you']:
        print("Bot: I am batman")
        break
    response = decode_sequence(user_input)
    print("Bot:", response)


# In[16]:


pip install h5py --upgrade


# In[9]:


model.save('chatbot_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


# In[ ]:




