import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Step 1: Create folder and save training data
# =========================
folder_name = "TrainingFolder"
training_file = os.path.join(folder_name, "TrainingData.txt")

# Sample text data
sample_texts = [
    "Hello, how can i help you today?",
    "i can't assist with unknown english words",
    "i don't have Full Pre-trained Data today, so i can't use math equations",
    "it's a math equations in Full Training Data, so it's {equations of math}"
]

# Create folder if not exists
os.makedirs(folder_name, exist_ok=True)

# Save training data
with open(training_file, "w", encoding="utf-8") as f:
    for line in sample_texts:
        f.write(line + "\n")

print(f"Training data saved to: {training_file}")

# =========================
# Step 2: Create vocabulary
# =========================
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sample_texts)

# Save vocabulary to JSON
vocab_file = os.path.join(folder_name, "vocab.json")
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)

print(f"Vocabulary saved to: {vocab_file}")
print("Vocabulary:", tokenizer.word_index)

# =========================
# Step 3: Prepare sequences
# =========================
sequences = tokenizer.texts_to_sequences(sample_texts)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# =========================
# Step 4: Build simple LSTM model
# =========================
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
    LSTM(32),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# =========================
# Step 5: Dummy training example
# =========================
# For demonstration, we'll train on predicting next word (simplified)
X = padded_sequences[:, :-1]  # input
y = padded_sequences[:, -1]   # target

model.fit(X, y, epochs=10, verbose=1)

print("Training complete.")
