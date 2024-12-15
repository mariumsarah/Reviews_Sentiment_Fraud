#!/usr/bin/env python3
# pip install selenium
# pip install seaborn
# pip install textblob
from nltk.tokenize import word_tokenize
#Install necessary libraries
#!pip install nltk pandas numpy scikit-learn
#Importing  libraries
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#Downloading  NLTK datasets for preprocessing of data(reviews)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
"""
Script Name: review_analysis.py
Description: Web Scrape business Reviews
Author: Sarah Marium
Date: Dec 2024
"""
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
#stratifiedSamling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

plt.style.use('ggplot')

import nltk

#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)


# Preprocess text function
def PreProcessText(review):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(review.lower())
    # Remove stopwords and non-alphanumeric tokens, and lemmatize(converting it in to its base form)
        # Initialize Lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')  # Ensure 'not' is not removed
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

    # Function to predict sentiment
def predict_sentiment(review,vectorizer,grid_search):
    review_processed = PreProcessText(review)
    review_vectorized = vectorizer.transform([review_processed])
    prediction = grid_search.predict(review_vectorized)
    return prediction[0]

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def main(): 
    
    # TO OUTPUT
    df_reviews = pd.read_csv("output.csv")

    # -------------------------------------------------------------------
    # GITHUB: https://github.com/SayamAlt/Fake-Reviews-Detection/tree/main
    # -------------------------------------------------------------------

    df = pd.read_csv('Preprocessed Fake Reviews Detection Dataset.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.dropna(inplace=True)
    df['length'] = df['text_'].apply(len)

    df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_

    review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)

    # pipeline = Pipeline([
    #     ('bow',CountVectorizer(analyzer=text_process)),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',MultinomialNB())
    # ])

    # predictions = pipeline.predict(review_test)
    # predictions

    # pipeline.fit(review_train,label_train)

    # print('Classification Report:',classification_report(label_test,predictions))
    # print('Confusion Matrix:',confusion_matrix(label_test,predictions))
    # print('Accuracy Score:',accuracy_score(label_test,predictions))

    # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')


    # pipeline = Pipeline([
    #     ('bow',CountVectorizer(analyzer=text_process)),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',RandomForestClassifier())
    # ])

    # pipeline.fit(review_train,label_train)


    # rfc_pred = pipeline.predict(review_test)
    # rfc_pred

    # print('Classification Report:',classification_report(label_test,rfc_pred))
    # print('Confusion Matrix:',confusion_matrix(label_test,rfc_pred))
    # print('Accuracy Score:',accuracy_score(label_test,rfc_pred))
    # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')


    # pipeline = Pipeline([
    #     ('bow',CountVectorizer(analyzer=text_process)),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',DecisionTreeClassifier())
    # ])
    # pipeline.fit(review_train,label_train)

    # dtree_pred = pipeline.predict(review_test)
    # dtree_pred

    # print('Classification Report:',classification_report(label_test,dtree_pred))
    # print('Confusion Matrix:',confusion_matrix(label_test,dtree_pred))
    # print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
    # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')

    # pipeline = Pipeline([
    #     ('bow',CountVectorizer(analyzer=text_process)),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',KNeighborsClassifier(n_neighbors=2))
    # ])

    # pipeline.fit(review_train,label_train)

    # knn_pred = pipeline.predict(review_test)
    # knn_pred
    # print('Classification Report:',classification_report(label_test,knn_pred))
    # print('Confusion Matrix:',confusion_matrix(label_test,knn_pred))
    # print('Accuracy Score:',accuracy_score(label_test,knn_pred))
    # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')

    # pipeline = Pipeline([
    #     ('bow',CountVectorizer(analyzer=text_process)),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC())
    # ])

    # pipeline.fit(review_train,label_train)

    # svc_pred = pipeline.predict(review_test)
    # svc_pred

    # print('Classification Report:',classification_report(label_test,svc_pred))
    # print('Confusion Matrix:',confusion_matrix(label_test,svc_pred))
    # print('Accuracy Score:',accuracy_score(label_test,svc_pred))
    # print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')

    pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',LogisticRegression())
    ])


    pipeline.fit(review_train,label_train)

    lr_pred = pipeline.predict(review_test)
    lr_pred

    #print('Classification Report:',classification_report(label_test,lr_pred))
    #print('Confusion Matrix:',confusion_matrix(label_test,lr_pred))
    print('Accuracy Score:',accuracy_score(label_test,lr_pred))
    print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')

    #print('Performance of various ML models:')
    #print('\n')
    #print('Logistic Regression Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')
    # print('K Nearest Neighbors Prediction Accuracy:',str(np.round(accuracy_score(label_test,knn_pred)*100,2)) + '%')
    # print('Decision Tree Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
    # print('Random Forests Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')
    # print('Support Vector Machines Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')
    # print('Multinomial Naive Bayes Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')
    
    df_reviews['New Detection']= pipeline.predict(df_reviews['Review'])

    df_reviews.to_csv('output_fraud1.csv', index=False)

    # -------------------------------------------------------------------
    # GITHUB: https://github.com/RaoMubashir760/Fraud-App-Detection-using-sentiment-analysis-By-Rao-Mubashir/tree/main
    # -------------------------------------------------------------------

    # Load dataset
    df = pd.read_csv('DatasetReviewsAndSentiments.csv')

    # Vectorize the text data with bi-grams and tri-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Consider unigrams, bi-grams, and tri-grams
    X = vectorizer.fit_transform(df['Review'])
    y = df['Label']
    print(y.value_counts())
    # Split data into training and testing sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    # Define the model
    model = MultinomialNB()

    # Define the parameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 0.7],
        'fit_prior': [True, False]
    }

    #Set up GridSearchCV with stratified k-fold cross-validation
    stratified_kfold = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, cv=stratified_kfold, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    #Best parameters found by using GridSearchCV
    #print(f"Best parameters: {grid_search.best_params_}")

    #Now project will make predictions
    y_pred = grid_search.predict(X_test)

    # Evaluate the model using different metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # print(f"Confusion Matrix:\n{cm}")

    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1_score}")

    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

        # Read a CSV file into a DataFrame
    df_reviews['Fraud Detection'] = ''
    # Input by user to detect the fraud app
    for index, row in df_reviews.iterrows():
        predicted_sentiment = predict_sentiment(row['Review'],vectorizer,grid_search)
        #print(f"Review: {row['Review']}")
        #print(f"Predicted sentiment for the review: {predicted_sentiment}")
        if predicted_sentiment == "negative":
            df_reviews.at[index,'Fraud Detection'] = "App is Fraud"
        else:
            df_reviews.at[index,'Fraud Detection'] = "App is Not Fraud"

    # -------------------------------------------------------------------
    # GITHUB: https://github.com/kntb0107/fake_review_detector/tree/main
    # -------------------------------------------------------------------
    #LOAD PICKLE FILES
    model = pickle.load(open('best_model.pkl','rb')) 
    vectorizer = pickle.load(open('count_vectorizer.pkl','rb')) 

    df_reviews['Fraud Review Detector'] = ''
   
    for index, row in df_reviews.iterrows():
        
        cleaned_review = text_preprocessing(row['Review'])
        process = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(process)
        p = ''.join(str(i) for i in prediction)
    
        if p == 'True':
            df_reviews.at[index,'Fraud Review Detector'] = "The review entered is Legitimate."
        if p == 'False':
            df_reviews.at[index,'Fraud Review Detector'] = "The review entered is Fraudulent."

    # -------------------------------------------------------------------
    # KAGGLE: https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial
    # -------------------------------------------------------------------
    # Read in data
    sia = SentimentIntensityAnalyzer()
    df_reviews['SIA NEG'] = ''
    df_reviews['SIA NEU'] = ''
    df_reviews['SIA POS'] = ''
   
    for index, row in df_reviews.iterrows():
        sias =  sia.polarity_scores(row['Review'])
        df_reviews.at[index,'SIA NEG'] =sias['neg']
        df_reviews.at[index,'SIA NEU'] = sias['neu']
        df_reviews.at[index,'SIA POS'] = sias['pos']
        
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    df_reviews['twitter NEG'] = ''
    df_reviews['twitter NEU'] = ''
    df_reviews['twitter POS'] = ''

    for index, row in df_reviews.iterrows():
        encoded_text = tokenizer(row['Review'], return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        df_reviews.at[index,'twitter NEG'] =scores[0]
        df_reviews.at[index,'twitter NEU'] = scores[1]
        df_reviews.at[index,'twitter POS'] =scores[2]

    df_reviews.to_csv('output_fraud4.csv', index=False)

if __name__ == "__main__":
    main()


# #     with open("data.json", "r") as file:
# #         retrieved_dict = json.load(file)
# #     all_links = retrieved_dict['organic_results']
# #     link_found_bbb = ''
# #     for x in all_links: 
# #         if 'www.bbb.org' in x['link']:
# #             link_found_bbb = x['link']
# #     print("BBB Link found "+link_found_bbb)

# #     driver=webdriver.Firefox()
# #     source=0
# #     driver.get("https://www.bbb.org/us/wa/seattle/profile/ecommerce/amazoncom-1296-7039385/complaints")
# # time.sleep(5)