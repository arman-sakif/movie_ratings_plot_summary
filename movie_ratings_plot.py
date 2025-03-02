import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path to dataset
path = "../cleaned_movie_dataset_v2.csv"  # Adjust this to your local path

# Read data into dataframe
df = pd.read_csv(path)

# Create 'text' column by concatenating 'title' and 'plot'
df['text'] = df['title'] + ' ' + df['plot']

# Create a new DataFrame 'final_df' with only 'text' and 'rating_class' columns
final_df = df[['text', 'rating_class']].copy()

# Rename "rating_class" to "label" in final_df
final_df = final_df.rename(columns={'rating_class': 'label'})

# Check the shape and first few rows of final_df
print(final_df.shape)
print(final_df.head())

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to format time into hh:mm:ss
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=3,  # Assuming 3 classes for classification
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

# Tokenizing the text data
max_len = 0
max_len_itr = 0

for i in range(len(final_df)):
    sent = final_df['text'][i]
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    leng = len(input_ids)
    if(leng > max_len):
        max_len = leng
        max_len_itr = i

print('Max sentence length: ', max_len)
print('Sentence: ', final_df['text'][max_len_itr])

label_list = final_df['label'].tolist()
input_ids = []
attention_masks = []

# Tokenize all sentences and add padding/truncation
for sent in final_df['text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode
                        add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
                        max_length=512,            # Pad & truncate all sentences
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        truncation=True,
                        return_tensors='pt',       # Return pytorch tensors
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(label_list)

# Set seed for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Define k-fold cross-validation
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed_val)

# Prepare dataset
dataset = TensorDataset(input_ids, attention_masks, labels)
dataset_size = len(dataset)
print(f"Total dataset size: {dataset_size}")

# Initialize variables to track results
total_conf_matrix = np.zeros((3, 3))  # Assuming 3 classes
total_y_true = []
total_y_pred = []

# Training and validation loop
batch_size = 16
epochs = 4
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n======== Fold {fold + 1} / {k_folds} =======")

    # Create train and validation subsets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_subset, sampler=RandomSampler(train_subset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_subset, sampler=SequentialSampler(val_subset), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training phase
    for epoch_i in range(epochs):
        print(f"\nEpoch {epoch_i + 1} / {epochs} - Fold {fold + 1}")
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.2f}")

    # Validation phase
    print("\nRunning Validation...")
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions = np.argmax(logits, axis=1)

        y_true.extend(label_ids)
        y_pred.extend(predictions)
        y_probs.extend(torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy())

    # Store results across folds
    total_y_true.extend(y_true)
    total_y_pred.extend(y_pred)
    total_conf_matrix += confusion_matrix(y_true, y_pred)

    # Print fold results
    print("Fold Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Fold ROC-AUC Score: {roc_auc_score(y_true, y_probs, multi_class='ovr'):.4f}")

# Final evaluation across all folds
print("\n======== Final Evaluation Across All Folds ========")
print("Overall Classification Report:")
print(classification_report(total_y_true, total_y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(total_conf_matrix.astype(int), annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Across All Folds")
plt.show()
