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
import numpy as np

class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

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
    doc = ' '.join(filtered_tokens)
    return doc



if __name__ == "__main__":
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')


    ### Consolidate Depression and Suicide files into one dataframe with 2 columnns (label, text)
    ### Ensured 50/50 breakdown between the 2.
    ### Need a third
    df = pd.DataFrame(columns=["label", "text"])
    i = 0
    for file in os.listdir("./data/slighly less uncleaned/Suicide"):
        text = open(("data/slighly less uncleaned/Suicide/" + file)).read()
        df.loc[i] = ["suicide"] + [text]
        i += 1

    j = 0
    while j < i:
        file = os.listdir("./data/slighly less uncleaned/Depression")[j]
        print(file)
        text = open(("data/slighly less uncleaned/Depression/" + file)).read()
        df.loc[i + j] = ["depression"] + [text]
        j += 1

    ## We know that this dataset is collected from actual surveys of participants.
    ## thus we can safely disregard syntactic or semantic structures of the text.

    text_values = df["text"]
    print(df)


    ## we do some simple normalization of the text data.

    i = 0
    while i < 2*j:
        text = df.loc[i]['text']
        text = normalize_document(text)
        df.loc[i]['text'] = text
        i += 1
    print(df)
    print(df.info())

#######################################################################################################################
    # ### unary classifer
    #
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import f1_score
    # from sklearn.svm import OneClassSVM
    #
    # # split into train/test sets
    # trainX, testX, trainy, testy = train_test_split(df['text'], df['label'], test_size=0.3, shuffle=True)
    #
    # # encode text data
    # Encoder = sklearn.preprocessing.LabelEncoder()
    # trainy = Encoder.fit_transform(trainy)
    # testy = Encoder.fit_transform(testy)
    #
    # # Vectorization of the words, we will use TF-IDF
    #
    # Tfidf_vect = sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000)
    # Tfidf_vect.fit(df['text'])
    # Train_X_Tfidf = Tfidf_vect.transform(trainX)
    # Test_X_Tfidf = Tfidf_vect.transform(testX)
    #
    # trainX = Train_X_Tfidf
    # testX = Test_X_Tfidf
    #
    # # define outlier detection model
    # model = OneClassSVM(gamma='scale', nu=0.01)
    # # fit on majority class
    # trainX = trainX[trainy == 0]
    # model.fit(trainX)
    # # detect outliers in the test set
    # yhat = model.predict(testX)
    # print("SVM Accuracy Score -> ", sklearn.metrics.accuracy_score(yhat, testy) * 100)
    # # mark inliers 1, outliers -1
    # testy[testy == 1] = -1
    # testy[testy == 0] = 1
    # # calculate score
    # score = f1_score(testy, yhat, pos_label=-1)
    # print('F1 Score: %.3f' % score)

#######################################################################################################################
    ### binary classifier

    ## Split the data

    Train_X, Test_X, Train_Y, Test_Y = sklearn.model_selection  .train_test_split(df['text'], df['label'],
                                                                        test_size=0.3, shuffle = True)
    # encode text data
    Encoder = sklearn.preprocessing.LabelEncoder()
    Encoder_Test = sklearn.preprocessing.OrdinalEncoder
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    # Vectorization of the words, we will use TF-IDF

    Tfidf_vect = sklearn.feature_extraction.text.TfidfVectorizer(max_features=10000)
    Tfidf_vect.fit(df['text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    print(Train_X_Tfidf)
    print(len(Train_Y))


    # We shall try the binary classifier first.
    # fit the training dataset on the classifier
    SVM = sklearn.svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", sklearn.metrics.accuracy_score(predictions_SVM, Test_Y) * 100)

#######################################################################################################################

    # WIP

    # from sklearn.feature_extraction.text import TfidfVectorizer
    # user_input = ["I wanna die"]
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(user_input)
    # fitted_user_input = vectorizer.transform([user_input[0]])
    # print(fitted_user_input)
    #
    # result = SVM.predict(fitted_user_input)
    # print(result)

