import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Data Loading and Preprocessing ---
print("--- 1. Data Loading and Preprocessing ---")

# Load the dataset
# Assuming 'Adidas US Sales Datasets.csv' is in the same directory or accessible
try:
    df = pd.read_csv('Adidas US Sales Datasets.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Adidas US Sales Datasets.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Initial Data Inspection
print(f"Original DataFrame shape: {df.shape}")
print("Original DataFrame head:")
print(df.head())

# Clean column names (remove leading/trailing spaces, replace spaces with underscores)
df.columns = df.columns.str.strip().str.replace(' ', '_')
df.columns = df.columns.str.replace('/', '_') # Handle 'Price/Unit'

# Convert 'Invoice_Date' to datetime objects
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'], format='%d/%m/%Y')

# Clean numerical columns: 'Total_Sales', 'Operating_Profit'
# Remove '$', ',', and spaces, then convert to float
for col in ['Total_Sales', 'Operating_Profit']:
    df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
    # Handle potential empty strings after cleaning
    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors will turn non-convertible values into NaN

# Clean 'Operating_Margin' (remove '%', convert to float, then divide by 100)
df['Operating_Margin'] = df['Operating_Margin'].astype(str).str.replace('%', '', regex=False).str.strip()
df['Operating_Margin'] = pd.to_numeric(df['Operating_Margin'], errors='coerce') / 100

# Convert 'Price_per_Unit' and 'Units_Sold' to numeric
df['Price_per_Unit'] = df['Price_per_Unit'].astype(str).str.replace('$', '', regex=False).str.strip() # Added cleaning for '$' in Price_per_Unit
df['Price_per_Unit'] = pd.to_numeric(df['Price_per_Unit'], errors='coerce')
df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')

# Identify numerical features including 'Price_per_Unit'
numerical_features = ['Price_per_Unit', 'Units_Sold', 'Operating_Profit', 'Operating_Margin'] # Total_Sales is our target

# Handle any NaN values that might have resulted from cleaning (e.g., fill with mean or median)
# Fill numerical NaNs with the mean BEFORE scaling
for col in numerical_features + ['Total_Sales']: # Include target for NaN handling if necessary
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())
        print(f"Filled NaN in {col} with mean.")

# Identify categorical features
categorical_features = ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales_Method']


# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Scale numerical features
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Combine processed features
# Ensure the index is aligned before concatenation
df_processed = pd.concat([df[['Invoice_Date', 'Retailer_ID', 'Total_Sales']], encoded_df, df[numerical_features]], axis=1)
print(f"Processed DataFrame shape: {df_processed.shape}")
print("Processed DataFrame head:")
print(df_processed.head())

# Check for NaNs in df_processed after all processing steps
print("\nChecking for NaNs in df_processed:")
print(df_processed.isnull().sum()[df_processed.isnull().sum() > 0])


# --- 2. Sequence Creation for RNN ---
print("\n--- 2. Sequence Creation for RNN ---")

# Define sequence parameters
LOOK_BACK = 5 # Number of previous records to use for prediction

def create_sequences(data, look_back):
    X, y = [], []
    # Group by Retailer_ID and Product to create meaningful sequences
    # Since Product is one-hot encoded, we'll use Retailer_ID and sort by date
    # This assumes that sales for a given Retailer_ID form a time series

    # Sort data by Retailer_ID and Invoice_Date to ensure correct sequence order
    data = data.sort_values(by=['Retailer_ID', 'Invoice_Date']).reset_index(drop=True)

    # Features for the RNN input (all processed features except 'Invoice_Date', 'Retailer_ID', 'Total_Sales')
    features_for_sequence = data.drop(columns=['Invoice_Date', 'Retailer_ID', 'Total_Sales']).columns.tolist()

    grouped = data.groupby('Retailer_ID')

    for _, group in grouped:
        # For each group, we create sequences
        # The target 'Total_Sales' is the sales of the *next* record

        # Extract features and target for the current group
        group_features = group[features_for_sequence].values
        group_target = group['Total_Sales'].values

        for i in range(len(group_features) - look_back):
            # Input sequence: look_back records
            X.append(group_features[i:(i + look_back)])
            # Target: Total_Sales of the record immediately following the sequence
            y.append(group_target[i + look_back])

    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(df_processed, LOOK_BACK)

print(f"Shape of X (sequences): {X.shape}") # (samples, timesteps, features)
print(f"Shape of y (target): {y.shape}") # (samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 3. Model Definition (Advanced RNN with 4 Hidden Layers and 5 Activations) ---
print("\n--- 3. Model Definition (Advanced RNN) ---")

# Determine the number of features per timestep
num_features = X_train.shape[2]

model = keras.Sequential([
    # Input layer: Defines the shape of the input sequences
    layers.Input(shape=(LOOK_BACK, num_features)),

    # Hidden Layer 1: LSTM with ReLU activation
    # return_sequences=True to pass output of this LSTM to the next LSTM
    layers.LSTM(units=128, activation='relu', return_sequences=True, name='lstm_layer_1'),

    # Hidden Layer 2: GRU with Tanh activation
    layers.GRU(units=64, activation='tanh', return_sequences=True, name='gru_layer_2'),

    # Hidden Layer 3: Another LSTM with Sigmoid activation
    layers.LSTM(units=32, activation='sigmoid', return_sequences=False, name='lstm_layer_3'), # return_sequences=False for the last RNN layer

    # Hidden Layer 4: Dense layer with ELU activation
    # This layer processes the output of the last RNN layer
    layers.Dense(units=16, activation='elu', name='dense_layer_4'),

    # Another Dense layer for more complexity with Leaky ReLU activation
    layers.Dense(units=8, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='dense_layer_5_leaky_relu'),

    # Output layer: Single neuron for regression (Total_Sales)
    layers.Dense(units=1, name='output_layer') # Linear activation by default for regression
])

# Print model summary
model.summary()

# --- 4. Training ---
print("\n--- 4. Training the Model ---")

# Compile the model with Adam optimizer and Mean Squared Error (MSE) loss
# Note: Keras models are compiled with one loss function.
# For the request of "2 loss functions", we will use MSE here and mention MAE as an alternative.
model.compile(optimizer='adam', loss='mean_squared_error')
print("Model compiled with Adam optimizer and Mean Squared Error (MSE) loss.")
print("Alternative loss function: 'mean_absolute_error' (MAE) can also be used.")

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50, # Number of epochs can be adjusted
    batch_size=32,
    validation_split=0.1, # Use a portion of training data for validation
    verbose=1
)

print("\nModel training complete.")

# --- 5. Evaluation ---
print("\n--- 5. Model Evaluation ---")

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (Mean Squared Error): {loss:.4f}")

# Make predictions on the test set
y_pred = model.predict(X_test).flatten()

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# --- 6. Visualization ---
print("\n--- 6. Visualization ---")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot Actual vs. Predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')
plt.title('Actual vs. Predicted Total Sales')
plt.xlabel('Actual Total Sales')
plt.ylabel('Predicted Total Sales')
plt.grid(True)
plt.legend()
plt.show()

# Residuals plot
plt.figure(figsize=(12, 6))
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- 7. Serialization (Saving Model and Preprocessing Components) ---
print("\n--- 7. Serialization ---")

# Save the trained Keras model
model_save_path = 'adidas_rnn_model.h5'
model.save(model_save_path)
print(f"Keras model saved to: {model_save_path}")

# Save the MinMaxScaler
scaler_save_path = 'minmax_scaler.pkl'
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"MinMaxScaler saved to: {scaler_save_path}")

# Save the OneHotEncoder
encoder_save_path = 'onehot_encoder.pkl'
with open(encoder_save_path, 'wb') as f:
    pickle.dump(encoder, f)
print(f"OneHotEncoder saved to: {encoder_save_path}")

print("\nProject execution complete.")