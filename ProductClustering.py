import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import torch

# Load CSV file
df = pd.read_csv("BigBasket Products.csv")
# df = df.head(100)  # Limit to first 1000 records for faster processing

# Concatenate relevant columns into a single text field
df["combined_text"] = df[["product", "category", "sub_category", "type", "description"]].astype(str).agg(" ".join, axis=1)

# Sample 80% of the data for the validation set and 20% for the training set
validation_df = df.sample(frac=0.2, random_state=42)
training_df = df.drop(validation_df.index)

# Print the number of rows and columns
print(f"Training dataframe shape: {training_df.shape}")
print(f"Validation dataframe shape: {validation_df.shape}")

# Load SBERT model with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"

# List the available GPUs
if device == "cuda":
    print(f"Found {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Direct torch to use the first GPU
if device == "cuda":
    torch.cuda.set_device(0)

    # The selected GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found, using CPU")

# Print the progress every 100 records
def convert_text_embedding(text, idx):
    if idx % 100 == 0:
        print(f"Processed {idx} records for text embeddings")
    return model.encode(text).tolist()

model = SentenceTransformer("all-mpnet-base-v2", device=device)  # Small, efficient model

print(f"Using device: {device}")
print("Generating embeddings of combined text...")

# Load the combined text embeddings dictionary if it exists
product_combined_text_embeddings_map = {}
try:
    with open("product_combined_text_embeddings.json", "r") as f:
        print("Loading combined text embeddings...")
        product_combined_text_embeddings_map = json.load(f)
except FileNotFoundError:
    # Create a dictionary of product IDs and their corresponding combined text embeddings
    print("Generating combined text embeddings...")

    product_combined_text_embeddings_map = {row["index"]: convert_text_embedding(row["combined_text"], idx) for idx, row in df.iterrows()}
    
    # Save the text embeddings dictionary to a JSON file
    print("Saving combined text embeddings...")
    with open("product_combined_text_embeddings.json", "w") as f:
        json.dump(product_combined_text_embeddings_map, f)

print("Generating embeddings of description...")

# Generate embeddings for the "description" column
product_description_embeddings_map = {}
try:
    with open("product_description_embeddings.json", "r") as f:
        print("Loading description embeddings...")
        product_description_embeddings_map = json.load(f)
except FileNotFoundError:
    # Create a dictionary of product IDs and their corresponding description embeddings
    print("Generating description embeddings...")
    product_description_embeddings_map = {row["index"]: convert_text_embedding(row["description"] if row["description"] is None else "", idx) for idx, row in df.iterrows()}
    
    # Save the description embeddings dictionary to a JSON file
    print("Saving description embeddings...")
    with open("product_description_embeddings.json", "w") as f:
        json.dump(product_description_embeddings_map, f)


print("Fitting combined text to tokenizer model...")

# Tokenization using TF-IDF to extract feature words
vectorizer = TfidfVectorizer(stop_words="english", max_features=30)  # Limit features for efficiency
vectorizer.fit(training_df["combined_text"])  # Fit TF-IDF on full dataset

# Load the product tokens dictionary if it exists
token_analyzer = vectorizer.build_analyzer()
def tokenize_text(text, idx):
    # Function to tokenize text using TF-IDF
    if idx % 100 == 0:
        print(f"Processed {idx} records for tokenization")
    return token_analyzer(text)

product_feature_tokens = {}
try:
    with open("product_feature_tokens.json", "r") as f:
        print("Loading product feature tokens...")
        product_feature_tokens = json.load(f)
except FileNotFoundError:
    # Create a dictionary of product IDs and their corresponding feature tokens
    print("Tokenizing product text...")
    product_feature_tokens = {row["index"]: tokenize_text(row["combined_text"], idx) for idx, row in df.iterrows()}
    
    # Save the product feature tokens dictionary to a JSON file
    print("Saving product feature tokens...")
    with open("product_feature_tokens.json", "w") as f:
        json.dump(product_feature_tokens, f)

# Generate embeddings for each token
# Load the token embeddings dictionary if it exists
product_token_embeddings = {}
token_embedding_dictionary = {}
try:
    with open("token_embedding_dictionary.json", "r") as f:
        print("Loading token embeddings...")
        token_embedding_dictionary = json.load(f)

        print(f"Token count: {len(token_embedding_dictionary)}")

        for k, v in product_feature_tokens.items():
            product_token_embeddings[k] = [np.asarray(token_embedding_dictionary[t], dtype=np.float32) for t in v]
except FileNotFoundError:
    print("Generating token embeddings...")
    token_embedding_dictionary = {}

    def generate_embeddings(tokens, index):
        embeddings = []
        for t in tokens:
            if t not in token_embedding_dictionary:
                token_embedding_dictionary[t] = model.encode(t, device=device).tolist()
            embeddings.append(token_embedding_dictionary[t])
            
        if index % 10 == 0:
            print(f"Processed {index} records for token embeddings")
        return embeddings
    
    # Create a dictionary of product IDs and tokens generated from product_feature_tokens
    for idx, (k, v) in enumerate(product_feature_tokens.items()):
        product_token_embeddings[k] = np.asarray(generate_embeddings(v, idx), dtype=np.float32)

    # Save the token embedding dictionary to a JSON file
    print("Saving token embeddings...")
    with open("token_embedding_dictionary.json", "w") as f:
        json.dump(token_embedding_dictionary, f)



# Save the tokenized features and embeddings to a JSON file
product_token_embeddings_aggregated = {}
try:
    with open("product_aggregated_token_embeddings.json", "r") as f:
        print("Loading aggregated product token embeddings...")
        product_token_embeddings_aggregated = json.load(f)
except FileNotFoundError:
    # Aggregate token embeddings into a single vector (mean pooling)
    print("Aggregating token embeddings...")
    product_token_embeddings_aggregated = {k: np.mean(v, axis=0).tolist() for k, v in product_token_embeddings.items()}
    df["aggregated_token_embedding"] = df["index"].map(product_token_embeddings_aggregated)

    print("Saving the product tokenized aggregated embeddints...")
    with open("product_aggregated_token_embeddings.json", "w") as f:
        json.dump(product_token_embeddings_aggregated, f)


print("Saved embeddings and tokenized features to products_with_tokenized_embeddings.json")