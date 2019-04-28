# Disaster Response Pipeline Project
### Libraries
* pandas==0.23.1
* SQLAlchemy==1.1.13
* nltk==3.2.4
* sklearn=0.20.1
* lightgbm=2.2.2
### Project Structure
```
|--app/  
|   |--templates                        # html template for Flask
|   |--run.py                           # Flask app entry point
|--data/
|   |--disaster_categories.csv          # categories data
|   |--disaster_messages.csv            # messages data
|   |--process_data.py                  # data processing script
|--models/  
|   |--train_classifier.py              # model training script
|--notebooks/  
|   |--ETL-pipeline-preparation.ipynb   # experiment notebook for process_data.py
|   |--ML-pipeline-preparation.ipynb    # experiment notebook for train_classifier.py
|
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
