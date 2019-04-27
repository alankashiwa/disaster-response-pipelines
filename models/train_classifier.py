import sys
import os
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(database_filepath):
    """ Load data from sqlite database

    Args:
        database_filepath: path of db file
    
    Returns:
        X: the messages feature
        Y: 36 output labels
        categroy_names: category names for y labels

    """

    # Read database into pandas dataframe
    engine = create_engine(os.path.join('sqlite:///',database_filepath))
    df = pd.read_sql_table('message', engine) 

    # Features & Labels
    s_X = df['message']
    df_y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return s_X.values, df_y.values, df_y.columns

def tokenize(text):
    """Tokenization function for CountVectorizer

    Args:
        text: raw text string to be tokenized
    
    Returns:
        tokenized word list

    """
    # Obtain tokens
    tokens = word_tokenize(text)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Clean tokens to be return
    clean_tokens = []
    
    # Stop words 
    stop_words = set(stopwords.words('english'))
    
    # Loop through and return preprocess tokens
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        if clean_token not in stop_words:
            clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    """ Pipeline and GridSearch model builder

    Returns:
        GridSearchCV model object with parameters set

    """

    # Pipeline structure
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(lgb.LGBMClassifier()))])
    
    # Parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': [(1,1), (1,2)],
        'vect__max_df': [1.0, 0.7],
        'clf__estimator__bagging_fraction': [1.0, 0.8]
    }

    # GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters,  verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Make predictions and evaluate model performance

    Args:
        model: classifier to evaluate
        X_test: test features 
        Y_test: test labels
        categories_names: category names for y labels

    Returns:
        None. The results will be printed in the console

    """

    # Make prediction
    Y_pred = model.predict(X_test)
    # Generate reports
    for ith, cate in enumerate(category_names):
        accuracy = round(accuracy_score(Y_test[:,ith], Y_pred[:,ith]), 2)
        print(cate.upper(), 'accuracy: {}'.format(accuracy))
        print(classification_report(Y_test[:,ith], Y_pred[:,ith]))


def save_model(model, model_filepath):
    """ Save the tuned model as pickle

    Args:
        model: model to be saved
        model_filepath: saved pickle file path
    
    Returns:
        None. 
    
    """
    with open(model_filepath, 'wb') as fp:
        pickle.dump(model, fp)


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