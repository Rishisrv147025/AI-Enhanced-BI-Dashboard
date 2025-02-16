import pandas as pd
import plotly.express as px
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# List of chart types that the model can predict
chart_labels = [
    "histogram", "scatter", "lineplot", "bar", "boxplot", "heatmap", "piechart", 
    "areachart", "violin", "sunburst", "treemap", "funnel", "density_heatmap", "density_contour", "clustered_column"
]

# Enhanced Text Preprocessing
def preprocess_text(text):
    """Preprocesses input text by removing stop words, punctuation, and lemmatizing."""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

def extract_entities(text):
    """Extracts entities from the text that might be relevant for chart generation."""
    doc = nlp(text.lower())
    return [ent.text for ent in doc.ents]

# Enhanced Feature Extraction
def extract_ngrams(tokens, n=2):
    """Extracts n-grams from the tokenized text."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def extract_pos_tags(tokens):
    """Extracts POS tags from the tokenized text."""
    doc = nlp(' '.join(tokens))
    return [token.pos_ for token in doc]

# Enhanced Dataset Preparation
class EnhancedChartDataset(Dataset):
    """Enhanced dataset with n-grams and POS tags."""
    def __init__(self, prompts, labels, vocab, max_len=50):
        self.prompts = prompts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        tokens = preprocess_text(prompt)
        ngrams = extract_ngrams(tokens)
        pos_tags = extract_pos_tags(tokens)
        all_tokens = tokens + ngrams + pos_tags
        indexed_tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in all_tokens]
        indexed_tokens = indexed_tokens[:self.max_len] + [self.vocab["<PAD>"]] * (self.max_len - len(indexed_tokens))
        return torch.tensor(indexed_tokens), torch.tensor(label)

# Enhanced Model Architecture
class EnhancedTextClassificationModel(nn.Module):
    """Enhanced model with attention and convolutional layers."""
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=15, dropout=0.5):
        super(EnhancedTextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Reshape for Conv1d
        conv_out = self.conv1(embedded)
        conv_out = conv_out.permute(0, 2, 1)  # Reshape back for LSTM
        lstm_out, (hn, cn) = self.lstm(conv_out)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.fc(self.dropout(attended))
        return out

# Enhanced Training Loop
def train_enhanced_model(train_dataset, vocab_size):
    """Trains the enhanced model using the training dataset."""
    model = EnhancedTextClassificationModel(vocab_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model.train()
    best_loss = float('inf')
    patience = 3
    for epoch in range(10):  # Train for 10 epochs
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                break
    return model

# Enhanced Chart Generation
def generate_chart(action, df, user_columns=None, highlight_values=None, entities=None):
    """Generates charts based on the specified action, data, and extracted entities."""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # If user specifies columns, use them; otherwise, use the first available columns
    if user_columns:
        x_col = user_columns.get("x", None)
        y_col = user_columns.get("y", None)
    else:
        x_col, y_col = None, None

    # Use extracted entities to suggest columns
    if entities:
        for entity in entities:
            if entity in numerical_cols:
                x_col = x_col or entity
            elif entity in categorical_cols:
                y_col = y_col or entity

    # Dynamically select columns based on the chart type
    if action == "histogram":
        if len(numerical_cols) > 0:
            if not x_col:
                x_col = numerical_cols[0]  # Default to first numerical column
            fig = px.histogram(df, x=x_col)
            fig.update_layout(title="Histogram")

    elif action == "scatter":
        if len(numerical_cols) > 1:
            if not x_col:
                x_col = numerical_cols[0]
            if not y_col:
                y_col = numerical_cols[1]
            fig = px.scatter(df, x=x_col, y=y_col)
            fig.update_layout(title="Scatter Plot")

    elif action == "lineplot":
        if len(numerical_cols) > 1:
            if not x_col:
                x_col = numerical_cols[0]
            if not y_col:
                y_col = numerical_cols[1]
            fig = px.line(df, x=x_col, y=y_col)
            fig.update_layout(title="Line Plot")

    elif action == "bar":
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            if not x_col:
                x_col = categorical_cols[0]
            if not y_col:
                y_col = numerical_cols[0]
            fig = px.bar(df, x=x_col, y=y_col)
            fig.update_layout(title="Bar Plot")

    elif action == "boxplot" and len(numerical_cols) > 0:
        fig = px.box(df, y=numerical_cols[0])
        fig.update_layout(title="Box Plot")

    elif action == "heatmap" and len(numerical_cols) > 1:
        fig = px.imshow(df.corr(), text_auto=True)
        fig.update_layout(title="Heatmap")

    elif action == "piechart" and len(categorical_cols) > 0:
        fig = px.pie(df, names=categorical_cols[0])
        fig.update_layout(title="Pie Chart")

    elif action == "areachart" and len(numerical_cols) > 0:
        fig = px.area(df, x=x_col, y=y_col if len(numerical_cols) > 1 else numerical_cols[0])
        fig.update_layout(title="Area Chart")

    elif action == "violin" and len(numerical_cols) > 0:
        fig = px.violin(df, y=y_col)
        fig.update_layout(title="Violin Plot")

    elif action == "sunburst" and len(categorical_cols) > 1:
        fig = px.sunburst(df, path=[categorical_cols[0], categorical_cols[1]])
        fig.update_layout(title="Sunburst Chart")

    elif action == "treemap" and len(categorical_cols) > 1:
        fig = px.treemap(df, path=[categorical_cols[0], categorical_cols[1]])
        fig.update_layout(title="Treemap")

    elif action == "funnel" and len(categorical_cols) > 0 and len(numerical_cols) > 0:
        fig = px.funnel(df, x=categorical_cols[0], y=numerical_cols[0])
        fig.update_layout(title="Funnel Chart")

    elif action == "density_heatmap" and len(numerical_cols) > 1:
        fig = px.density_heatmap(df, x=x_col, y=y_col)
        fig.update_layout(title="Density Heatmap")

    elif action == "density_contour" and len(numerical_cols) > 1:
        fig = px.density_contour(df, x=x_col, y=y_col)
        fig.update_layout(title="Density Contour")

    elif action == "clustered_column" and len(categorical_cols) > 0 and len(numerical_cols) > 1:
        fig = px.bar(df, x=x_col, y=y_col)
        fig.update_layout(title="Clustered Column Chart")

# Prepare Dataset
def prepare_dataset():
    """Prepares dataset for training."""
    prompts = [
        "show me a histogram of sales", "scatter plot of sales vs time", "plot a line chart for revenue", 
        "bar chart of categories", "boxplot of age distribution", "heatmap of correlation", 
        "pie chart for market share", "area chart for sales trends", "violin plot of prices", 
        "sunburst chart for hierarchy", "treemap for expenditures", "funnel chart for conversion rates", 
        "density heatmap of data", "density contour for variable distribution", "clustered column chart"
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Corrected labels, 0 to 14
    all_tokens = [token for prompt in prompts for token in preprocess_text(prompt)]
    vocab = {token: idx + 1 for idx, token in enumerate(set(all_tokens))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)
    train_prompts, test_prompts, train_labels, test_labels = train_test_split(prompts, labels, test_size=0.2)
    train_dataset = EnhancedChartDataset(train_prompts, train_labels, vocab)
    test_dataset = EnhancedChartDataset(test_prompts, test_labels, vocab)
    return train_dataset, test_dataset, vocab
