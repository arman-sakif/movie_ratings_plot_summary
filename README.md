# movie_ratings_plot_summary

This project demonstrates training a BERT model for sequence classification using 3-fold cross-validation. The code is designed to run in a Python environment.

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Setup

1. Clone the repository or download the code.
2. Install the required packages:

```sh
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

## Code Overview
### Data Preparation
- Load the dataset from a CSV file.
- Concatenate the title and plot columns to create a text column.
- Rename the rating_class column to label.
### Model and Tokenizer
- Load the BERT tokenizer and model from the Hugging Face library.
- Configure the model for sequence classification with 3 output labels.
### Training
- Perform 3-fold cross-validation.
- Train the model using the AdamW optimizer and a linear learning rate scheduler.
Evaluate the model on the validation set after each epoch.
### Evaluation
- Calculate and print the classification report and ROC-AUC score for each fold.
- Visualize the confusion matrix across all folds.
## Results
The script prints the training and validation loss, accuracy, and time for each epoch and fold. It also provides a final evaluation across all folds, including a confusion matrix visualization.