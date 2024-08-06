# -*- coding: utf-8 -*-
"""

@author: Xin Wang
"""


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, UpSampling1D, BatchNormalization, Dropout, TimeDistributed
from keras.models import Model, Sequential
import h5py
import tensorflow as tf
from keras.callbacks import EarlyStopping

# Load X data from MATLAB .mat file
with h5py.File('X_Len2_Step001Full.mat', 'r') as file:
    X = file['newX'][:]
# Load Y data from MATLAB .mat file
with h5py.File('Y_Len2_Step001Full.mat', 'r') as file:
    Y = file['newY'][:]
scaler = StandardScaler()

# Transpose the data to have samples as the first dimension
X = np.transpose(X, (1, 0))
y = np.transpose(Y, (1, 0))

# Print shapes to verify
print(f"X shape: {X.shape}, y shape: {y.shape}")

# # Standardize the data
# num_features = X.shape[2]
# time_steps = X.shape[1]
# X = scaler.fit_transform(X.reshape(-1, num_features)).reshape(-1, time_steps, num_features)
X = X.reshape((X.shape[0], X.shape[1], 1))


# Calculate the sizes for each split
total_samples = X.shape[0]
train_size = int(0.6 * total_samples)
valid_size = int(0.2 * total_samples)

# Split the data
X_train_scaled = X[:train_size, :, :]
X_val_scaled = X[train_size:train_size + valid_size, :, :]
X_test_scaled = X[train_size + valid_size:, :, :]
y_train = y[:train_size]
y_val = y[train_size:train_size + valid_size, :]
y_test = y[train_size + valid_size:, :]





from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Flatten, Dense
# Assuming X_train_scaled has already been defined
input_shape = (X_train_scaled.shape[1], 1)  # Make sure to adjust this based on your actual data

# Define the input tensor
input_tensor = Input(shape=input_shape)

# First convolutional layer
conv1 = Conv1D(64, 3, activation='relu', padding='same')(input_tensor)
bn1 = BatchNormalization()(conv1)
mp1 = MaxPooling1D(2)(bn1)
drop1 = Dropout(0.2)(mp1)

# Second convolutional layer
conv2 = Conv1D(128, 3, activation='relu', padding='same')(drop1)
bn2 = BatchNormalization()(conv2)
mp2 = MaxPooling1D(2)(bn2)
drop2 = Dropout(0.2)(mp2)

# Second convolutional layer
conv2 = Conv1D(256, 3, activation='relu', padding='same')(drop2)
bn2 = BatchNormalization()(conv2)
mp2 = MaxPooling1D(2)(bn2)
drop2 = Dropout(0.2)(mp2)

# LSTM layer
lstm_layer = LSTM(128, return_sequences=True)(drop2)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
lstm_layer = LSTM(128)(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)
# Flattening the data
# lstm_layer = Flatten()(lstm_layer)

# Dense layer for output
dense1 = Dense(256, activation='relu')(lstm_layer)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation='relu')(dense1)
drop4 = Dropout(0.2)(dense1)
output_layer = Dense(1, activation='sigmoid')(drop4)  # Binary output
# Create the model
model = Model(inputs=input_tensor, outputs=output_layer)

model.summary()
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Adding callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

# Addressing the imbalance in the dataset
weights = {0: 1, 1: 10}  # Assuming class 0 is more frequent

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_val_scaled, y_val),
    epochs=100, 
    batch_size=64*4,
    class_weight=weights,  # Using class weights to handle imbalance
    callbacks=callbacks
)


my_th = 0.5

# Predict for training data
train_predictions = model.predict(X_train_scaled)
train_predicted_classes = (train_predictions > my_th).astype(int)

# Predict for validation data
val_predictions = model.predict(X_val_scaled)
val_predicted_classes = (val_predictions > my_th).astype(int)

# Predict for test data
test_predictions = model.predict(X_test_scaled)
test_predicted_classes = (test_predictions > my_th).astype(int)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, predictions, predicted_classes):
    accuracy = accuracy_score(y_true, predicted_classes)
    precision = precision_score(y_true, predicted_classes)
    recall = recall_score(y_true, predicted_classes)
    f1 = f1_score(y_true, predicted_classes)
    auc = roc_auc_score(y_true, predictions)  # AUC uses the probabilities, not the binary predictions
    return accuracy, precision, recall, f1, auc

# Metrics for the training dataset
train_accuracy, train_precision, train_recall, train_f1, train_auc = calculate_metrics(y_train, train_predictions, train_predicted_classes)

# Metrics for the validation dataset
val_accuracy, val_precision, val_recall, val_f1, val_auc = calculate_metrics(y_val, val_predictions, val_predicted_classes)

# Metrics for the test dataset
test_accuracy, test_precision, test_recall, test_f1, test_auc = calculate_metrics(y_test, test_predictions, test_predicted_classes)

# Print all metrics
print("Training Metrics:")
print(f"Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1 Score: {train_f1}, AUC: {train_auc}")

print("\nValidation Metrics:")
print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}, AUC: {val_auc}")

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}, AUC: {test_auc}")











# # # Create an LSTM Model for Regression
# # lstm_model = Sequential()
# # lstm_model.add(LSTM(256, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))

# # lstm_model.add(LSTM(256, return_sequences=True))
# # lstm_model.add(LSTM(256, return_sequences=True))
# # lstm_model.add(LSTM(24))
# # # lstm_model.add(Dense(256, activation='linear'))
# # # lstm_model.add(Dense(24, activation='linear'))

# # Model construction
# lstm_model = Sequential([
#     LSTM(256, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True),
#     Dropout(0.2),
#     LSTM(256, return_sequences=True),
#     Dropout(0.2),
#     LSTM(256, return_sequences=True),
#     Dropout(0.2),
#     # LSTM(256, return_sequences=False),
#     # Dropout(0.2),
#     # Dense(256, activation='relu'),
#     # Dense(24, activation='relu')  # Assuming 24 is the output dimension for regression
#     TimeDistributed(Dense(1, activation='linear'))
# ])




# # Compile the model
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

# # Print the model summary
# lstm_model.summary()

# # Set up GPU growth (optional but recommended)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# # Train the model with Early Stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = lstm_model.fit(X_train_scaled, y_train,
#                           epochs=1000, batch_size=256, shuffle=True,
#                           validation_data=(X_val_scaled, y_val),
#                           callbacks=[early_stopping])
# lstm_model.save('lstm_model.h5')

# from keras.models import load_model
# lstm_model = load_model('lstm_model.h5')


# # Make predictions on the test set
# y_pred = lstm_model.predict(X_test_scaled)

# # Evaluate the predictions
# mse = np.mean((y_pred - y_test) ** 2)
# print(f"Mean Squared Error on Test Set: {mse}")

# import matplotlib.pyplot as plt
# # Plot predictions vs actual values for the test set
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.flatten(), label='Actual Values')
# plt.plot(y_pred.flatten(), label='Predicted Values', alpha=0.7)
# plt.legend()
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.title('Predicted vs Actual Values on Test Set')
# plt.show()





# # Make predictions on the train, validation, and test sets
# y_train_pred = lstm_model.predict(X_train_scaled)
# y_val_pred = lstm_model.predict(X_val_scaled)
# y_test_pred = lstm_model.predict(X_test_scaled)

# # Calculate MSE for each set
# mse_train = np.mean((y_train_pred - y_train) ** 2)
# mse_val = np.mean((y_val_pred - y_val) ** 2)
# mse_test = np.mean((y_test_pred - y_test) ** 2)

# print(f"Mean Squared Error on Training Set: {mse_train}")
# print(f"Mean Squared Error on Validation Set: {mse_val}")
# print(f"Mean Squared Error on Test Set: {mse_test}")

# # Plot predictions vs actual values for the test set
# plt.figure(figsize=(10, 6))
# plt.plot(y_test.flatten(), label='Actual Values')
# plt.plot(y_test_pred.flatten(), label='Predicted Values', alpha=0.7)
# plt.legend()
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.title('Predicted vs Actual Values on Test Set')
# plt.show()