# Import necessary libraries
from Functions import *
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Mapping from labels to integers
label_map = {label: num for num, label in enumerate(actions)}

# Prepare sequences and labels from dataset
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            filepath = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(filepath)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences and labels to numpy arrays and categorical data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard callback setup
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

# Calculate and print average accuracy over all epochs
average_accuracy = np.mean(history.history['categorical_accuracy'])
print(f"Average Accuracy over all epochs: {average_accuracy * 100:.2f}%")

# Evaluate the model on the test data and display accuracy as percentage
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print model summary
model.summary()

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model
model.save('model.h5')
