# -*- coding: utf-8 -*-
"""MR - training BERT V[0.3].py"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Prep
path = os.path.join("..", "dataset", "cleaned_movie_dataset_v2.csv")

# rating classes:
# 0: x<5
# 1: x<7
# 2: x>7

df = pd.read_csv(path)
df.head()

# Create the 'text' column by concatenating 'title' and 'plot'
df['text'] = df['title'] + ' ' + df['plot']

# Create a new DataFrame 'final_df' with only the 'text' and 'rating_class' columns
final_df = df[['text', 'rating_class']].copy()

# Rename "rating_class" to "label" in final_df
final_df = final_df.rename(columns={'rating_class': 'label'})

print(final_df.shape)
final_df.head()

# Function Definitions
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load Transformer Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

model = BertForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 3, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.to(device)

# Model Training
max_len = 0
max_len_itr = 0

for i in range(len(final_df)):
    sent = final_df['text'][i]
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    # Update the maximum sentence length.
    leng = len(input_ids)
    if(leng > max_len):
        max_len = leng
        max_len_itr = i

print('Max sentence length: ', max_len)
print('sentence: ', final_df['text'][max_len_itr])

label_list = final_df['label'].tolist()  # Directly convert to list

input_ids = []
attention_masks = []

for sent in final_df['text']:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation = True,
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Append to lists
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Concatenate tensors outside the loop
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

batch_size = 16
epochs = 4
training_stats = []
total_t0 = time.time()

y_true_all = []
y_pred_all = []
y_scores_all = []

dataset = TensorDataset(input_ids, attention_masks, labels)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n======== Fold {fold + 1} / {k_folds} ========")

    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_subsampler, sampler=RandomSampler(train_subsampler), batch_size=batch_size)
    validation_dataloader = DataLoader(val_subsampler, sampler=SequentialSampler(val_subsampler), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch_i in range(epochs):
        print(f"\nEpoch {epoch_i + 1} / {epochs} - Fold {fold + 1}")

        # Training Phase
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.')

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
        training_time = format_time(time.time() - t0)

        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        # Validation Phase
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        y_true = []
        y_pred = []
        y_scores = []

        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions = np.argmax(logits, axis=1)

            total_eval_accuracy += flat_accuracy(logits, label_ids)
            y_true.extend(label_ids)
            y_pred.extend(predictions)
            y_scores.extend(logits[:, 1])  # Assuming binary classification

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        training_stats.append({
            'Fold': fold + 1,
            'Epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_scores)

# Final Evaluation
print("\n======== Final Evaluation Across All Folds ========")
print("Overall Classification Report:")
print(classification_report(y_true_all, y_pred_all))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true_all, y_pred_all), annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Across All Folds")
plt.show()
