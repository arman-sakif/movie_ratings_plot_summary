# -*- coding: utf-8 -*-
"""Merging CMU and our IMDb datasets."""

import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

"""
Creates a pandas DataFrame from a text file containing movie IDs and plot summaries.

Args:
  filename: The path to the text file.

Returns:
  A pandas DataFrame with two columns: 'movie_id' and 'plot'.
"""
def create_movie_dataframe(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            try:
                movie_id, plot = line.strip().split('\t', 1)  # Split by first tab
                data.append({'movie_id': movie_id, 'plot': plot})
            except ValueError:
                print(f"Error parsing line: {line}")  # Handle lines without a tab

    df = pd.DataFrame(data)
    return df

"""
Loads movie metadata from a TSV file into a pandas DataFrame.

Args:
  filename: The path to the TSV file.

Returns:
  A pandas DataFrame containing the movie metadata.
"""
def load_movie_metadata(filename):
    # Read the TSV file into a DataFrame
    df = pd.read_table(filename,
                       names=['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name',
                              'Release_date', 'Box_office_revenue', 'Runtime',
                              'Languages', 'Countries', 'Genres'],
                       sep='\t')

    # Convert relevant columns to appropriate data types
    df['Release_date'] = pd.to_datetime(df['Release_date'], errors='coerce')
    df['Box_office_revenue'] = pd.to_numeric(df['Box_office_revenue'], errors='coerce')
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')

    return df

"""
Merges two DataFrames based on 'movie_id' and 'Wikipedia_movie_ID'.

Args:
  df: The first DataFrame.
  movie_df: The second DataFrame.

Returns:
  A merged DataFrame.
"""
def merge_dataframes(df, movie_df):
    merged_df = pd.merge(df, movie_df,
                         left_on='movie_id',
                         right_on='Wikipedia_movie_ID',
                         how='left')
    return merged_df

def merge_dataframes_prioritize_merged_df(df1, merged_df):
    merged_result = pd.merge(df1, merged_df,
                             left_on='title',
                             right_on='Movie_name',
                             how='right')
    return merged_result

# Dataset path
path = os.path.join("..", "dataset")
print("Path to dataset files:", path)

# Load custom IMDb datasets
df1 = pd.read_csv(os.path.join(path, "movies_90s.csv"))
df2 = pd.read_csv(os.path.join(path, "movies_2000s.csv"))

# Concatenate the datasets
df = pd.concat([df1, df2], ignore_index=True)
df.tail()

df1 = df.copy()

df1.shape

df1.columns

# Rename 'Title' column to 'title'
df1 = df1.rename(columns={'Title': 'title'})

# Load plot summaries dataset
filename = os.path.join(path, 'plot_summaries.txt')
df = create_movie_dataframe(filename)
df.head()

df.columns

# Load movie metadata
filename = os.path.join(path, 'movie.metadata.tsv')
movie_df = load_movie_metadata(filename)
movie_df.head()

df.shape

movie_df.shape

# Convert 'movie_id' and 'Wikipedia_movie_ID' to string for merging
df['movie_id'] = df['movie_id'].astype(str)
movie_df['Wikipedia_movie_ID'] = movie_df['Wikipedia_movie_ID'].astype(str)

# Merge dataframes
merged_df = merge_dataframes(df, movie_df)
merged_df.head()

merged_df.shape

# Merge with custom IMDb dataset
merged_result = merge_dataframes_prioritize_merged_df(df1, merged_df)
merged_result.head()

merged_result.shape

merged_result.columns

# Create a new DataFrame with selected columns
temp = merged_result[['title', 'Year', 'Rating', 'plot', 'Genres']]

# Check for null values in each column
print(temp.isnull().sum())

# Drop rows with null values in key columns
temp = temp.dropna(subset=['title', 'Year', 'Rating', 'Genres'])

# Check for null values again
print(temp.isnull().sum())

# Check if DataFrame has any null values
if temp.isnull().values.any():
    print(temp[temp.isnull().any(axis=1)])

# Create final DataFrame
final_df = temp.copy()

final_df.shape

# Check for duplicate titles
duplicate_titles = final_df[final_df.duplicated(subset=['title'])]
num_duplicates = len(duplicate_titles)
print(f"Number of duplicate titles: {num_duplicates}")

# Drop duplicate titles
final_df = final_df.drop_duplicates(subset=['title'])
print(f"Shape after removing duplicates: {final_df.shape}")

# Rename columns
final_df = final_df.rename(columns={'Year': 'year', 'Rating': 'rating', 'Genres': 'genres'})

# Reset index
final_df = final_df.reset_index(drop=True)

final_df.tail()

print(final_df.shape)

# Save the final DataFrame to a CSV file
final_df.to_csv(os.path.join(path, 'cmu_customIMDB.csv'), index=False)
