# movie_ratings_plot_summary

This repository provides a Python implementation of BERT (Bidirectional Encoder Representations from Transformers) model training for text classification tasks, using 3-fold cross-validation. The dataset consists of movie reviews with corresponding ratings, and the goal is to classify the reviews into three categories based on their ratings.

## Prerequisites

To run the code, make sure you have the following Python libraries installed:

- `transformers`: For using the BERT model and tokenizer.
- `torch`: For deep learning and PyTorch model handling.
- `scikit-learn`: For metrics and k-fold cross-validation.
- `matplotlib`: For data visualization.
- `seaborn`: For more attractive visualizations.

You can install the required dependencies using pip:

```bash
pip install transformers torch scikit-learn matplotlib seaborn
```

## Dataset

The dataset used for this project consists of multiple CSV files, each representing a different approach to collecting and organizing movie data. Below is a detailed breakdown of the datasets:

### Approach 1: Wikipedia + IMDb
- **File**: `cleaned_movie_dataset_v2.csv`  
  This dataset combines movie data from Wikipedia and IMDb. It has been cleaned and structured for ease of use.  

### Approach 2: Custom IMDb
This approach uses custom IMDb datasets divided by decades and a separate file for testing:
- **a) File**: `movies_90s.csv`  
  Contains movies released in the 1990s.  
- **b) File**: `movies_2000s.csv`  
  Contains movies released in the 2000s.  
- **c) File**: `movies_latest.csv`  
  Contains the latest movies and is **never used for training**. It is kept separate exclusively for testing purposes.  

### Approach 3: Wikipedia + Custom IMDb
- **File**: `cmu_customIMDB.csv`  
  This dataset merges Wikipedia data with custom IMDb data, providing a comprehensive collection of movie information.  

### Dataset Structure
All datasets are CSV files containing movie reviews, with at least the following columns:
- `title`: The title of the movie.  
- `plot`: The plot description of the movie.  
- `rating_class`: A classification of the movie's rating:  
  - `0`: Rating < 5  
  - `1`: Rating < 7  
  - `2`: Rating > 7  

Ensure your dataset is structured similarly, and update the path variable in the code to point to the correct location of your dataset CSV file.

## Workflow
The code performs the following steps:

### 1. Data Preparation
- Reads the dataset from the specified file.
- Creates a new column text by concatenating the title and plot columns.
- Renames rating_class to label for clarity.

### 2. Model Setup
- Loads a pre-trained BERT tokenizer and model (`bert-base-cased`).
- Configures the BERT model to handle three output labels for classification.

### 3. Tokenization and Data Processing
- Tokenizes the text data (concatenated title and plot) and applies padding and truncation to ensure uniform input lengths.
- Creates `input_ids`, `attention_masks`, and `labels` tensors for training.

### 4. K-Fold Cross-Validation
- Performs 3-fold cross-validation using KFold from scikit-learn.
- For each fold, the model is trained and validated, and evaluation metrics are computed (accuracy, classification report, ROC-AUC score, confusion matrix).

### 5. Training and Validation
- The model is trained using the AdamW optimizer, with a learning rate of 2e-5 and gradient clipping.
- After each epoch, the model is validated on the validation set, and performance metrics are logged.
- Training loss, validation loss, and accuracy are printed during the process.

### 6. Evaluation
- After training is completed for all folds, the final performance metrics are displayed, including the classification report and the confusion matrix across all folds.

## How to Run the Code
1. Clone this repository or download the script to your local machine.
2. Update the path variable in the code to point to the location of your CSV dataset file.
3. Run the script in your local Python environment:

```bash
python train_bert_model.py
```

The script will perform the following:

- Train the BERT model using 3-fold cross-validation.
- Output training and validation metrics for each fold.
- Display the final evaluation metrics, including a confusion matrix visualization.

## Example Output
After running the code, you should see output similar to this:

```bash
======== Fold 3 / 3 ========
Epoch 4 / 4 - Fold 3
  Batch 40 of 224. Elapsed: 0:00:59.
  Batch 80 of 224. Elapsed: 0:01:57.
  Batch 120 of 224. Elapsed: 0:02:56.
  Batch 160 of 224. Elapsed: 0:03:55.
  Batch 200 of 224. Elapsed: 0:04:53.
  Average training loss: 0.10
  Training epoch took: 0:05:28

Running Validation...
  Accuracy: 0.97
  Validation Loss: 0.10
  Validation took: 0:00:52

Fold Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.96      0.96       218
           1       0.96      0.99      0.97      1030
           2       0.98      0.91      0.94       543

    accuracy                           0.96      1791
   macro avg       0.96      0.95      0.96      1791
weighted avg       0.96      0.96      0.96      1791

Fold ROC-AUC Score: 0.9954

======== Final Evaluation Across All Folds ========
Overall Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.63      0.70       619
           1       0.78      0.90      0.84      3131
           2       0.79      0.62      0.69      1625

    accuracy                           0.78      5375
   macro avg       0.78      0.72      0.74      5375
weighted avg       0.78      0.78      0.78      5375

Confusion Matrix Across All Folds

```
## Visualizations
- **Confusion Matrix**: A heatmap that displays the performance of the classifier across all folds.

## Notes
- The model uses the `bert-base-cased` version of BERT for tokenization and sequence classification.
- You can adjust the number of epochs or batch size based on your hardware capabilities.
- The script is designed to run on either a GPU or CPU, depending on what is available.

## Experimental Results

## Approach 1
| Metric        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| **Macro Avg**   | 0.78      | 0.72   | 0.74     |
| **Weighted Avg**| 0.78      | 0.78   | 0.78     |

## Approach 2
| Metric        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| **Macro Avg**   | 0.79      | 0.77   | 0.78     |
| **Weighted Avg**| 0.81      | 0.81   | 0.81     |

## Approach 3
| Metric        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| **Macro Avg**   | 0.79      | 0.77   | 0.78     |
| **Weighted Avg**| 0.81      | 0.81   | 0.81     |


## License

