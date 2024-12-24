import pandas as pd
import torch
import torch.nn as nn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load Spacy's English model for better tokenization
nlp = spacy.load('en_core_web_sm')

# Advanced Tokenizer for better query understanding
class AdvancedTokenizer:
    def __init__(self):
        self.operations = [
            "max", "min", "sum", "difference", "top", "mean", "average",
            "groupby", "filter", "sort", "median", "count", "pivot",
            "join", "merge", "null", "range", "date", "trend", "normalize",
            "bin", "aggregate", "unique", "standardize"
        ]

    def tokenize(self, query):
        query = query.lower()
        doc = nlp(query)  # Use spacy's NLP model to process the query
        tokens = [token.text for token in doc if token.is_alpha or token.is_digit or token.text in ['>', '<', '=', '+', '-', '.', '/', '%']]  # Include operators
        operation = None
        columns = []
        conditions = []
        for token in tokens:
            if token in self.operations:
                operation = token
            elif token.isalpha():
                columns.append(token)
            elif token.isdigit():
                conditions.append(token)
            elif token in ['>', '<', '=', '+', '-', '%']:  # Include operators
                conditions.append(token)
        return operation, columns, conditions

# Initialize tokenizer instance
tokenizer = AdvancedTokenizer()

# Enhanced Neural Network without the Embedding layer
class EnhancedNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnhancedNLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.bn2 = nn.BatchNorm1d(output_size)  # Batch normalization
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Pass through fully connected layer
        x = self.bn1(x)
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        return x

# Function to predict operation from query
def predict_operation(query):
    operation, _, _ = tokenizer.tokenize(query)
    input_data = tfidf_vectorizer.transform([query]).toarray()  # Convert query to tf-idf vector
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert to tensor
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# Execute query based on the predicted operation
def execute_query(query, df):
    operation = predict_operation(query)
    action = ''
    result = None
    selected_columns = []

    # Extract columns from the query
    column_matches = re.findall(r"\b[A-Za-z_]+\b", query)
    for col in column_matches:
        if col in df.columns:
            selected_columns.append(col)

    # Perform operations
    if operation == 0:  # Max operation
        result = df[selected_columns].max().to_dict()
        action = f'Performed Max Operation on Columns: {", ".join(selected_columns)}'
    elif operation == 1:  # Top N operation
        top_n_matches = re.findall(r'\d+', query)
        if not top_n_matches:
            result = {"error": "No valid number found for 'top N' operation."}
        else:
            top_n = int(top_n_matches[0])  # Extract number from query
            result = df.nlargest(top_n, selected_columns[0]).to_dict(orient='records')
            action = f'Fetched Top {top_n} Records by {selected_columns[0]}'
    elif operation == 2:  # Sum operation
        result = df[selected_columns].sum().to_dict()
        action = f'Calculated Sum of Columns: {", ".join(selected_columns)}'

    return result, action
