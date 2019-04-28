import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Message and label example
    messages_categories = []
    df_sample = df.sample(3)
    for idx, row in df_sample.iterrows():
        message_category = {}
        message_category['message'] = row['message']
        cates = []
        for cate, cls_result in row.items():
            if cls_result == 1:
                cates.append(cate)
        message_category['categories'] = ' ,'.join(cates)
        messages_categories.append(message_category)

    # extract data needed for visuals
    
    # Graph#1: Genre Counts Bar Chart
    genre_counts = df.groupby('genre').count()['message'].sort_values()
    genre_names = list(genre_counts.index)
    
    # Graph#2: 
    df_cate = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_counts = df_cate.sum().sort_values(ascending=False)
    category_names = df_cate.columns

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_counts,
                    y=genre_names,
                    orientation = 'h'
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre",
                },
                'xaxis': {
                    'title': "Count"
                },
                'margin': dict(l=120,r=10,t=140,b=80)
            }
        },
         {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(color='rgb(221,130,79)')
                )
            ],
            'layout': {
                'title': 'Labels',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Label"
                },
                'margin': dict(l=120,r=10,t=140,b=80)
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, messages_categories=messages_categories)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()