import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

# Load Spacy's English model with custom patterns
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("merge_entities")

# Enhanced Tokenizer with semantic pattern matching
class AdvancedTokenizer:
    def __init__(self):
        self.operations = {
            "max": ["max", "maximum", "highest"],
            "min": ["min", "minimum", "lowest"],
            "sum": ["sum", "total", "add"],
            "average": ["average", "mean", "avg"],
            "top": ["top", "first", "best"],
            "filter": ["filter", "where", "having"],
            "sort": ["sort", "order", "arrange"],
            "groupby": ["group by", "categorize", "segment"],
            "count": ["count", "number of"],
            "unique": ["unique", "distinct"],
            "trend": ["trend", "pattern", "over time"],
            "pivot": ["pivot", "rotate", "transpose"]
        }
        
        self.condition_patterns = {
            'range': r'between (\d+) and (\d+)',
            'greater': r'greater than (\d+)',
            'less': r'less than (\d+)',
            'equals': r'equal to (\d+)'
        }

    def tokenize(self, query):
        doc = nlp(query.lower())
        tokens = [token.text for token in doc]
        entities = [ent.text for ent in doc.ents]
        
        operation = next((op for op, aliases in self.operations.items() if any(alias in tokens for alias in aliases)), None)
        columns = [ent for ent in entities if ent in self._detect_columns(tokens)]
        conditions = self._extract_conditions(query)
        
        return operation, columns, conditions

    def _detect_columns(self, tokens):
        return [token for token in tokens if token.isalpha() and token not in self.operations]

    def _extract_conditions(self, query):
        conditions = {}
        for cond_type, pattern in self.condition_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                conditions[cond_type] = list(match.groups())
        return conditions

# Initialize enhanced tokenizer
tokenizer = AdvancedTokenizer()

# Fixed Neural Network with LayerNorm instead of BatchNorm
class EnhancedNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Changed to LayerNorm
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)  # Changed to LayerNorm
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.ln1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout2(x)
        return self.fc3(x)

# Enhanced query processing functions
def detect_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

def validate_columns(columns, df):
    valid_cols = [col for col in columns if col in df.columns]
    invalid_cols = [col for col in columns if col not in df.columns]
    return valid_cols, invalid_cols

def extract_aggregation_columns(query, df):
    doc = nlp(query)
    return [ent.text for ent in doc.ents if ent.text in df.columns]

# Enhanced training data with more examples
queries = [
    "What is the maximum sales value?",
    "Show the top 5 products by revenue",
    "Calculate total profit for last quarter",
    "Find the difference between product A and B sales",
    "Group sales data by region and category",
    "Filter records where price is greater than $100",
    "Sort customers by purchase history",
    "Pivot table showing sales per quarter",
    "Count number of unique customers",
    "Show sales trend over past 6 months",
    "Average rating per product category",
    "Minimum temperature recorded in January"
]

labels = ["max", "top", "sum", "difference", "groupby", 
         "filter", "sort", "pivot", "count", "trend", 
         "average", "min"]

# Preprocessing and model setup
tfidf_vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = tfidf_vectorizer.fit_transform(queries)
y_labels = list(range(len(labels)))

# Create DataLoader with batch size > 1
train_data = TensorDataset(
    torch.tensor(X_tfidf.toarray(), dtype=torch.float32),
    torch.tensor(y_labels, dtype=torch.long)
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize model, criterion, optimizer
model = EnhancedNLPModel(X_tfidf.shape[1], 256, len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop with proper mode handling
for epoch in range(20):
    model.train()  # Explicit training mode
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Enhanced prediction function with proper mode handling
def predict_operation(query):
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        input_tensor = torch.tensor(
            tfidf_vectorizer.transform([query]).toarray(),
            dtype=torch.float32
        )
        output = model(input_tensor)
        return torch.argmax(output).item()

# Enhanced query execution engine
def execute_query(query, df):
    try:
        operation_idx = predict_operation(query)
        operation = labels[operation_idx]
        df.columns = df.columns.str.lower()
        valid_cols, invalid_cols = validate_columns(
            extract_aggregation_columns(query, df), 
            df
        )
        
        if invalid_cols:
            return {"error": f"Invalid columns: {', '.join(invalid_cols)}"}, ""

        result = None
        action = f"Performed {operation} operation"
        
        # Handle numeric columns automatically
        numeric_cols = detect_numeric_columns(df)
        target_col = valid_cols[0] if valid_cols else numeric_cols[0] if numeric_cols else df.columns[0]
        
        if operation == "max":
            result = df[target_col].max()
        elif operation == "min":
            result = df[target_col].min()
        elif operation == "sum":
            result = df[target_col].sum()
        elif operation == "average":
            result = df[target_col].mean()
        elif operation == "top":
            n = int(re.search(r'\d+', query).group()) if re.search(r'\d+', query) else 5
            result = df.nlargest(n, target_col)
        elif operation == "filter":
            value = float(re.search(r'\d+', query).group()) if re.search(r'\d+', query) else None
            if value:
                result = df[df[target_col] > value]
            else:
                raise ValueError("No filter value found")
        elif operation == "sort":
            result = df.sort_values(by=target_col, ascending=False)
        elif operation == "groupby":
            group_col = valid_cols[0] if valid_cols else df.columns[0]
            result = df.groupby(group_col)[target_col].mean().reset_index()
        elif operation == "pivot":
            pivot_col = valid_cols[1] if len(valid_cols) > 1 else df.columns[1]
            result = df.pivot_table(values=target_col, index=df.columns[0], columns=pivot_col)
        elif operation == "count":
            result = df[target_col].nunique()
        elif operation == "trend":
            time_col = [col for col in df.columns if 'date' in col][0]
            result = df.set_index(time_col)[target_col].resample('M').mean()
        
        # Check if result is a DataFrame or Series for .to_dict()
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_dict(orient='records'), action
        else:
            return result, action
        
    except Exception as e:
        return {"error": str(e)}, "Error processing query"
