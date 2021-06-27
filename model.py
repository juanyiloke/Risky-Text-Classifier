# from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import decomposition, ensemble
#
# import pandas, xgboost, numpy, textblob, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

import pandas as pd
import os
import nltk
import re
import sklearn


def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    # doc = ' '.join(filtered_tokens)
    return doc



if __name__ == "__main__":

    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')


    ### Consolidate Depression files into one dataframe with 2 columnns (label, text)
    df = pd.DataFrame(columns=["label", "text"])
    i = 0
    for file in os.listdir("./data/slighly less uncleaned/Suicide"):
        text = open(("data/slighly less uncleaned/Suicide/" + file)).read()
        df.loc[i] = ["suicide"] + [text]
        i += 1

    ## We know that this dataset is collected from actual surveys of participants.
    ## thus we can safely disregard syntactic or semantic structures of the text.

    text_values = df["text"]
    print(text_values)


    ## we do some simple normalization of the text data.

    i = 0
    while i < 313:
        text = df.loc[i]['text']
        text = normalize_document(text)
        df.loc[i]['text'] = text
        i += 1
    print(df)
    print(df.info())


    ## Split the data

    Train_X, Test_X, Train_Y, Test_Y = sklearn.model_selection  .train_test_split(df['text'], df['label'],
                                                                        test_size=0.3)
    # encode text data
    Encoder = sklearn.preprocessing.LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Vectorization of the words, we will use TF-IDF

    Tfidf_vect = sklearn.feature_extraction.text.TfidfVectorizer(max_features=10000)
    Tfidf_vect.fit(df['text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    print(Train_X_Tfidf)
    print(len(Train_Y))


    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = sklearn.svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", sklearn.metrics.accuracy_score(predictions_SVM, Test_Y) * 100)
