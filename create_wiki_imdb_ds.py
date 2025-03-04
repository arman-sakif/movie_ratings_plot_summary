# Required Packages - numpy, pandas, kagglehub

import numpy as np
import pandas as pd

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

# part 1: load the dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shreyasur29/imdbratings")

print("Path to dataset files:", path)
df1 = pd.read_csv(path + '/IMDB-Ratings.csv', index_col=False)

print(df1.shape)

# this file has plot
filename = 'plot_summaries.txt'
df = create_movie_dataframe(filename)
print(df.head())

# this file has movie names
filename = 'movie.metadata.tsv'
movie_df = load_movie_metadata(filename)
print(df.shape)
print(movie_df.head())

df['movie_id'] = df['movie_id'].astype(str)
movie_df['Wikipedia_movie_ID'] = movie_df['Wikipedia_movie_ID'].astype(str)

# merge the two dataframes of CMU dataset
merged_df = merge_dataframes(df, movie_df)

# now merge CMU dataset with imdbRatings dataset
merged_result = merge_dataframes_prioritize_merged_df(df1, merged_df)

# drop rows where null values exist on the columns we care about
print(merged_result.columns)
temp = merged_result.dropna(subset=['title', 'averageRating', 'numVotes', 'Movie_name', 'plot', 'Genres', 'Countries', 'Languages'])
final_df = temp[['title', 'plot', 'Genres', 'Countries', 'Languages', 'averageRating', 'numVotes']].copy()

# save the final dataframe
print(final_df.shape)
print(final_df.head())
final_df.to_csv('movies.csv', index=False)
