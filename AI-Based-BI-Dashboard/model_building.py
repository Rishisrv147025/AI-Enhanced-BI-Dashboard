import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

# Function to handle data loading
def load_data(file_path):
    # Load CSV data into pandas DataFrame
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head())

    # Automatically detect date and target columns
    date_column = detect_date_column(df)
    target_column = detect_target_column(df, date_column)

    # If no target column is found, raise an error
    if target_column is None:
        raise ValueError("No valid numeric target column found in the dataset.")

    # If a date column is detected, process the date column
    if date_column:
        df = process_date_column(df, date_column)

    # Handle missing values (simple forward fill, but can be expanded)
    df.fillna(method='ffill', inplace=True)

    # Select only numeric features (excluding target and date columns)
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)  # Remove target column
    if date_column and date_column in numeric_features:
        numeric_features.remove(date_column)  # Remove date column if detected

    X = df[numeric_features].values
    y = df[target_column].values

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM [samples, time steps, features] if time-series, else flat for other tasks
    if date_column:
        X_scaled = np.expand_dims(X_scaled, axis=1)  # One time step per sample
    return X_scaled, y, target_column, date_column

# Function to detect the date column automatically
def detect_date_column(df):
    # Check for columns with date-like strings or datetime-like data types
    for column in df.columns:
        try:
            pd.to_datetime(df[column], errors='raise')
            return column  # If conversion is successful, it's likely the date column
        except:
            continue
    return None  # No date column detected

# Updated Function to detect the target column (assume it's a numeric column that is not a date)
def detect_target_column(df, date_column):
    # Drop the date column if detected
    if date_column:
        df = df.drop(columns=[date_column])

    # Search for the first numeric column as the target (excluding index and date column)
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and column != df.index.name and column != date_column:
            return column  # Return the first numeric column as the target

    return None  # No numeric column detected

# Add time-based features like year, month, week, etc.
def process_date_column(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['DayOfYear'] = df[date_column].dt.dayofyear
    df['Month'] = df[date_column].dt.month
    df['WeekOfYear'] = df[date_column].dt.isocalendar().week
    df['DayOfWeek'] = df[date_column].dt.weekday
    df = df.drop(columns=[date_column])  # Drop original date column
    return df

# Function to build the LSTM model for time-series forecasting
def build_lstm_model(input_dim, output_dim):
    inputs = layers.Input(shape=(1, input_dim))  # One time step per sample
    x = layers.LSTM(128, return_sequences=True)(inputs)  # Increased LSTM units
    x = layers.Dropout(0.3)(x)  # Increased dropout rate
    x = layers.LSTM(64)(x)  # More LSTM layers
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)  # Added dense layer
    x = layers.Dense(32, activation='relu')(x)  # Added more layers
    x = layers.Dense(output_dim)(x)  # Output layer
    model = models.Model(inputs, x)
    return model

# Function to build a Dense Neural Network (for non-time-series data)
def build_dense_model(input_dim, output_dim):
    model = models.Sequential([
        layers.InputLayer(input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),  # Increased dropout
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    return model

# Function to visualize future predictions and display them numerically
def plot_future_predictions(model, X_test, y_test, steps_ahead=10):
    last_known_data = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2]) if X_test.ndim > 2 else X_test[-1].reshape(1, -1)
    future_predictions = []

    for _ in range(steps_ahead):
        pred = model.predict(last_known_data)
        future_predictions.append(pred[0])
        last_known_data = np.roll(last_known_data, shift=-1, axis=1)
        last_known_data[0, -1, 0] = pred[0]  # Update the last time step with the prediction

    future_predictions = np.array(future_predictions).flatten()

    # Print future predictions numerically
    print("Future Predictions (Next {} steps):".format(steps_ahead))
    for i, pred in enumerate(future_predictions, start=1):
        print(f"Step {i}: {pred:.4f}")

    # Optionally, display the predictions in a DataFrame for better readability
    future_df = pd.DataFrame(future_predictions, columns=['Predicted Value'])
    print("\nFuture Predictions (Tabular View):")
    print(future_df)

    # Visualize the future predictions plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Data')
    plt.plot(np.arange(len(y_test), len(y_test) + len(future_predictions)), future_predictions, label='Future Predictions', color='orange')
    plt.title('Actual vs Future Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Function to evaluate model performance with multiple metrics
def evaluate_model(y_test, y_pred):
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_test, y_pred))

# Function to visualize prediction intervals
def plot_prediction_intervals(y_test, y_pred, alpha=0.05):
    lower_bound = y_pred - 1.96 * np.std(y_pred)
    upper_bound = y_pred + 1.96 * np.std(y_pred)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.fill_between(range(len(y_test)), lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Prediction Interval')
    plt.title('Prediction Intervals')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Function to plot monthly/weekly comparison
def plot_sales_comparison(y_test, y_pred, date_column):
    # Convert date to monthly or weekly data
    df = pd.DataFrame({'Date': date_column, 'Actual': y_test, 'Predicted': y_pred})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resample by month or week
    df_monthly = df.resample('M').mean()  # 'M' for monthly, 'W' for weekly
    df_monthly.plot(figsize=(10, 6))
    plt.title('Monthly Sales/Inventory Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

# Function to plot time series data
def plot_time_series(data, title='Time Series Data'):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.show()

# Function to plot cumulative sales/inventory
def plot_cumulative(data, title='Cumulative Sales/Inventory'):
    cumulative_data = np.cumsum(data)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_data)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Value')
    plt.show()

# Function to plot rolling mean and rolling std deviation
def plot_rolling_statistics(data, window=12):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std', color='green')
    plt.title(f'Rolling Mean & Standard Deviation (window={window})')
    plt.legend()
    plt.show()

# Function to compare forecast horizon
def plot_forecast_horizon(y_test, y_pred, steps_ahead=10):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label='True Values')
    plt.plot(range(len(y_test), len(y_test) + steps_ahead), y_pred[:steps_ahead], label=f'Forecast Horizon ({steps_ahead} steps)', color='orange')
    plt.title('Forecast Horizon Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Function for seasonality and trend decomposition
def plot_seasonality_trend_decomposition(data):
    decomposition = sm.tsa.seasonal_decompose(data, model='multiplicative', period=12)  # Assuming monthly data
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.show()

# Main function to load the dataset, train the model, and visualize results
def train_model(file_path):
    # Load and process data
    X_scaled, y, target_column, date_column = load_data(file_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build and compile the model
    input_dim = X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]
    output_dim = 1  # Predicting a single value (inventory, sales, etc.)

    if date_column:
        model = build_lstm_model(input_dim, output_dim)
    else:
        model = build_dense_model(input_dim, output_dim)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # EarlyStopping and ReduceLROnPlateau callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr], verbose=2)

    # Evaluate model performance
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')

    # Predictions
    y_pred = model.predict(X_test)

    # Visualizations
    plot_prediction_intervals(y_test, y_pred)
    plot_sales_comparison(y_test, y_pred, date_column)
    plot_time_series(y_test, title='Test Data Time Series')
    plot_cumulative(y_test, title='Cumulative Sales/Inventory')
    plot_rolling_statistics(y_test, window=12)
    plot_forecast_horizon(y_test, y_pred, steps_ahead=10)
    plot_seasonality_trend_decomposition(pd.Series(y_test))

    # Evaluate model performance with metrics
    evaluate_model(y_test, y_pred)

# Example usage
file_path = 'your_data_file.csv'  # Replace with the path to your CSV file
train_model(file_path)
