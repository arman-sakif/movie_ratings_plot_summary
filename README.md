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
The dataset used for this project should be a CSV file containing movie reviews, with at least two columns:

- `title`: The title of the movie.
- `plot`: The plot description of the movie.
- `rating_class`: A classification of the movie's rating (0: <5, 1: <7, 2: >7).

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
======== Fold 1 / 3 ========
Epoch 1 / 4 - Fold 1
  Average training loss: 0.45
  Training epoch took: 0:01:12

Running Validation...
  Accuracy: 0.82
  Validation Loss: 0.35
  Validation took: 0:00:12

Fold Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.75      0.76       100
           1       0.83      0.85      0.84       120
           2       0.79      0.75      0.77       110

    accuracy                           0.82       330
   macro avg       0.80      0.78      0.79       330
weighted avg       0.81      0.82      0.81       330

Fold ROC-AUC Score: 0.91

======== Final Evaluation Across All Folds ========
Overall Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.76      0.78       300
           1       0.84      0.85      0.85       360
           2       0.80      0.77      0.78       330

    accuracy                           0.81       990
   macro avg       0.81      0.79      0.80       990
weighted avg       0.81      0.81      0.81       990

Confusion Matrix Across All Folds

```
## Visualizations
- **Confusion Matrix**: A heatmap that displays the performance of the classifier across all folds.

## Notes
- The model uses the `bert-base-cased` version of BERT for tokenization and sequence classification.
- You can adjust the number of epochs or batch size based on your hardware capabilities.
- The script is designed to run on either a GPU or CPU, depending on what is available.

## License

