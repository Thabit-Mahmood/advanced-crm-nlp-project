# %% [markdown]
# # Tokenization

# %% [markdown]
# ## Import BERT Tokenizer and Tokenize the Data

# %%
import pandas as pd
from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to tokenize text
def tokenize_text(texts, tokenizer, max_length=512):
    """
    Tokenize a list of texts using the provided tokenizer.

    Parameters:
    - texts (list): List of text strings.
    - tokenizer (transformers.PreTrainedTokenizer): BERT tokenizer.
    - max_length (int): Maximum sequence length.

    Returns:
    - encodings (BatchEncoding): Tokenized texts.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# %% [markdown]
# ## Load the Preprocessed Data and Tokenize

# %%
# Load the preprocessed data
train_df = pd.read_csv('C:/Users/thabi/OneDrive/Documents/FYP/advanced-crm-nlp-project/data/clean_train_preprocessed.csv')
test_df = pd.read_csv('C:/Users/thabi/OneDrive/Documents/FYP/advanced-crm-nlp-project/data/clean_test_preprocessed.csv')

# Tokenize training and testing data
train_encodings = tokenize_text(train_df['clean_text'].tolist(), tokenizer)
test_encodings = tokenize_text(test_df['clean_text'].tolist(), tokenizer)

# %% [markdown]
# ## Inspect Tokenized Data

# %%
# Inspect the tokenized input_ids
print("Training Encodings - input_ids shape:", train_encodings['input_ids'].shape)
print("Testing Encodings - input_ids shape:", test_encodings['input_ids'].shape)

# Display a sample
print("Sample Training Input IDs:", train_encodings['input_ids'][0])
print("Sample Training Attention Mask:", train_encodings['attention_mask'][0])


