import pandas as pd
import ast

# Function to clean dictionary-like strings in columns
def clean_column(column_str):
    try:
        column_dict = ast.literal_eval(column_str)
        column_values = list(column_dict.values())
        return ', '.join(f'"{value}"' for value in column_values)
    except:
        return column_str  # Return original string if parsing fails

# Function to create a DataFrame from a text file containing movie IDs and plot summaries
def create_movie_dataframe(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            try:
                movie_id, plot = line.strip().split('\t', 1)  # Split by first tab
                data.append({'movie_id': movie_id, 'plot': plot})
            except ValueError:
                print(f"Error parsing line: {line}")  # Handle lines without a tab
    return pd.DataFrame(data)

# Function to load movie metadata from a TSV file
def load_movie_metadata(filename):
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

# Function to merge two DataFrames based on 'movie_id' and 'Wikipedia_movie_ID'
def merge_dataframes(df, movie_df):
    return pd.merge(df, movie_df,
                    left_on='movie_id',
                    right_on='Wikipedia_movie_ID',
                    how='left')

# Function to merge DataFrames with prioritization
def merge_dataframes_prioritize_merged_df(df1, merged_df):
    return pd.merge(df1, merged_df,
                    left_on='title',
                    right_on='Movie_name',
                    how='right')

# Main script
if __name__ == "__main__":
    # Load the IMDb ratings dataset
    import kagglehub
    path = kagglehub.dataset_download("shreyasur29/imdbratings")
    df1 = pd.read_csv(path + '/IMDB-Ratings.csv', index_col=False)

    # Load plot summaries and movie metadata
    plot_df = create_movie_dataframe('plot_summaries.txt')
    movie_metadata_df = load_movie_metadata('movie.metadata.tsv')

    # Merge the two DataFrames from the CMU dataset
    merged_df = merge_dataframes(plot_df, movie_metadata_df)

    # Merge the CMU dataset with the IMDb ratings dataset
    final_merged_df = merge_dataframes_prioritize_merged_df(df1, merged_df)

    # Clean the relevant columns
    for column in ['Genres', 'Countries', 'Languages', 'plot']:
        final_merged_df[column] = final_merged_df[column].apply(clean_column)

    # Drop rows with null values in the columns we care about
    final_df = final_merged_df.dropna(subset=['title', 'averageRating', 'numVotes', 'Movie_name', 'plot', 'Genres', 'Countries', 'Languages'])
    final_df = final_df[['title', 'plot', 'Genres', 'Countries', 'Languages', 'averageRating', 'numVotes']].copy()

    # Save the final DataFrame to a CSV file
    final_df.to_csv('movies.csv', index=False)
