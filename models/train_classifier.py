import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

# tokenizer
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# create model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# evaluate model
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

# save model
import pickle


def load_data(database_filepath):
    """Read database into dataframe and split data into feature and
    target columns.

    This particular implementation loads only the message column as
    model feature.

    Parameters
    ----------
    database_filepath : filepath. Filepath to sql storage.

    Returns:
    X : array_like. Feature column(s).
    y : array_like. Target columns.
    target_cols : array_like.  Target labels.
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_category', engine)

    # split by features and targets
    feature_cols = ['message']  # this would be changed if we want to add
                                # any more features to the model, such as genre
    target_cols = df.loc[:,'related':].columns.tolist()

    X = df[feature_cols].values.flatten()
    y = df[target_cols].values

    return X, y, target_cols


def tokenize(text):
    """Tokenize text to be used in functions such as CountVectorizer.

    Parameters
    ----------
    text : str. A line of text to be tokenized.

    Returns
    -------
    clean_tokens : array_like. Preprocessed tokens.
    """

    # remove punctuation and lowercase
    text = re.sub(r'[^\w\s]','', text).lower()
    # split into tokens
    tokens = word_tokenize(text)
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return clean_tokens


def build_model():
    """Build and return the predefined model pipeline for multi-label
    classification.
    """

    pipeline = Pipeline([
        ('features', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model's performance.

    Prints
    ------
    > The f1 score, precision and recall for each output category of
    the dataset.
    > The Hamming Loss for the model as a whole.
    """

    Y_pred = model.predict(X_test)

    print(f'Hamming Loss: {hamming_loss(Y_test, Y_pred)}')

    print(f'\nClassification Report for each category:\n')
    for i, name in enumerate(category_names):
        print(f'Category name: {name} \n \
             {classification_report(Y_test[:,i], Y_pred[:,i])}')



def save_model(model, model_filepath):
    """Save the model to the specified filepath.

    Parameters
    ----------
    model : trained model.
    model_filepath : filepath. Filepath specifying directory where to
    save and file name.
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
