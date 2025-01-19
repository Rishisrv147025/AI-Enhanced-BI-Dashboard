import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import io

# Data preprocessing functions

def detect_date_column(df):
    """
    Detects the column in the DataFrame that contains date information.
    Args:
        df: The DataFrame containing the data.
    Returns:
        The name of the date column, or None if not found.
    """
    for column in df.columns:
        try:
            pd.to_datetime(df[column], errors='raise')
            return column
        except Exception:
            continue
    return None

def detect_target_column(df, date_column):
    """
    Detects the target column for prediction in the DataFrame.
    Args:
        df: The DataFrame containing the data.
        date_column: The name of the date column to exclude from target detection.
    Returns:
        The name of the target column, or None if not found.
    """
    if date_column:
        df = df.drop(columns=[date_column])
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    return None

def process_date_column(df, date_column):
    """
    Creates time-based features from a date column.
    Args:
        df: The DataFrame containing the data.
        date_column: The name of the date column.
    Returns:
        The DataFrame with new time-based features and the date column removed.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['DayOfYear'] = df[date_column].dt.dayofyear
    df['Month'] = df[date_column].dt.month
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week
    df['DayOfWeek'] = df[date_column].dt.weekday
    return df.drop(columns=[date_column])

def load_data(file_path):
    """
    Loads and preprocesses the dataset from a CSV file.
    Args:
        file_path: The path to the CSV file.
    Returns:
        Scaled features, target values, and column names for target and date.
    """
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!\n", df.head())

    date_column = detect_date_column(df)
    target_column = detect_target_column(df, date_column)
    
    if not target_column:
        raise ValueError("No valid numeric target column found.")

    if date_column:
        df = process_date_column(df, date_column)

    df.fillna(method='ffill', inplace=True)
    
    numeric_features = [col for col in df.select_dtypes(include=np.number).columns if col != target_column]
    X = df[numeric_features].values
    y = df[target_column].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if date_column:
        X_scaled = np.expand_dims(X_scaled, axis=1)

    return X_scaled, y, target_column, date_column

# Model definitions

def build_lstm_model(input_dim, output_dim):
    """
    Builds and returns an LSTM model for time-series data.
    Args:
        input_dim: The number of features in the input data.
        output_dim: The number of outputs (usually 1 for regression tasks).
    Returns:
        A compiled LSTM model.
    """
    inputs = layers.Input(shape=(1, input_dim))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(output_dim)(x)
    return models.Model(inputs, x)

def build_dense_model(input_dim, output_dim):
    """
    Builds and returns a dense neural network model.
    Args:
        input_dim: The number of features in the input data.
        output_dim: The number of outputs (usually 1 for regression tasks).
    Returns:
        A compiled dense model.
    """
    model = models.Sequential([
        layers.InputLayer(input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    return model

# Future predictions plotting

def plot_future_predictions(model, X_test, steps_ahead=10):
    """
    Plots future predictions based on the trained model.
    Args:
        model: The trained model.
        X_test: The test features.
        steps_ahead: The number of future time steps to predict.
    """
    last_known_data = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2]) if X_test.ndim > 2 else X_test[-1].reshape(1, -1)
    future_predictions = []

    for _ in range(steps_ahead):
        pred = model.predict(last_known_data)
        future_predictions.append(pred[0])
        last_known_data = np.roll(last_known_data, shift=-1, axis=1)
        last_known_data[0, -1, 0] = pred[0]  # Update last time step

    future_predictions = np.array(future_predictions).flatten()
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Predicted Future Values')
    plt.xlabel('Time Step')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.show()

# Model training function

def train_model(file_path, forecast_steps, epochs, learning_rate):
    """
    Trains the model and plots future predictions.
    Args:
        file_path: The path to the CSV file.
        forecast_steps: The number of future steps to forecast.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
    """
    X_scaled, y, target_column, date_column = load_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]
    output_dim = 1  # Single output prediction

    if date_column:
        model = build_lstm_model(input_dim, output_dim)
    else:
        model = build_dense_model(input_dim, output_dim)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr], verbose=2)

    plot_future_predictions(model, X_test, steps_ahead=forecast_steps)
