import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Read and merge csv message and category files on id as common key and
    return as dataframe.
    
    For internal use only.
    
    Parameters
    ----------
    messages_filepath : filepath. Filepath to csv with messages
    categories_filepath: filepath. Filepath to csv with categories for each 
    message
    
    Returns
    -------
    df: pandas dataframe. Merged files as dataframe, on id as common key
    
    Notes
    -----
    CSV column signatures
    messages_filepath : 'id', 'message', 'original', 'genre'
    categories_filepath : 'id', 'categories'
    
    """
    
    # read in files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge files and drop duplicates
    df = messages.merge(categories, how='left', on='id').drop_duplicates(['id'])
    df.reset_index(drop=True, inplace=True)
    
    return df


def clean_data(df):
    """Clean data to be saved in database and used to train an ML
    classification model. This data must have been preprocessed through a
    call to load_data(messages_filepath, categories_filepath)
    
    For internal use only.
    
    Parameters
    ----------
    df : pandas dataframe. Each rown contains a message and category labels
    
    Returns
    -------
    df : pandas dataframe. Preprocessed data to be saved in database and 
    used for model training.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.rsplit(';', expand=True)
    
    # select the first row of the categories dataframe to get column names
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to 0,1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    # some entries are larger than 1, so we clip them    
    categories.clip(upper=1, inplace=True)
    
    # drop the original categories column from `df`
    # and concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1).drop(['categories'], axis=1)
    
    return df


def save_data(df, database_filename):
    """Save pandas dataframe into sql database by creating a new table named
    `message_category` 
    
    Parameters
    ----------
    df : pandas dataframe. Dataframe to be saved into database
    database_filename : filepath. Filepath to database. If it does not exist
    it will be created.
    
    Notes
    -----
    There is no return value, it just modifies the database.
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    
    with engine.connect() as connection:
        connection.execute('DROP TABLE IF EXISTS message_category')
        
    df.to_sql('message_category', engine, index=False)
      


def main():
    """Run the ETL pipeline to prepare data for classification.
    Usage found in the README file.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()