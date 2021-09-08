"""
pip install azure-ai-textanalytics
"""

import os
from secret import ta_endpoint, ta_key
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


class RiskClassifier:
    # Risky words, seperated by commas // TODO: Add more risky keywords
    risky_words_raw = "death,suicide,depressed,die,hurt,harm"

    def __init__(self, endpoint, key):
        self.ta_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.risky_words = RiskClassifier.risky_words_raw.lower().split(',')

    def isRisky(self, text, confidence_thresh=0, neg_thresh=0.5, verbose=False):
        """Returns whether or not text is risky.
        The check is composed of two portions: a keyword search and sentiment analysis.
        1. Keyword search involves checking for words that are risky (e.g. death, suicide, etc.)
            If keyword found, then it's immediately considered as risky
        2. Sentiment analysis is done by TextAnalytics. A text will then be considered risky
            if its sentiment is "negative" based on TextAnalytics

        Args:
            text (string or list[string]): Text (or list of texts) to be analyzed.
            confidence_thresh (float, optional): Minimum confidence in a negative sentiment. 
                Defaults to 0.
            neg_thresh (float, optional): Minimum fraction of sentences in a text to have
                negative sentiment before considering the text as risky. Defaults to 0.5.

        Returns:
            boolean: Whether or not the text is considered risky.
        """
        def isSentenceRisky(sentence):
            return sentence.sentiment == 'negative' and sentence.confidence_scores['negative'] > confidence_thresh

        def _print(string): 
            if verbose: 
                print(" > {}".format(string))
  
        # 1. Keyword Search ================================
        # Search for keywords in each of the text.
        # If found, it's automatically risky
        isKeyWordFound = any(word in text.lower() for word in self.risky_words)
        if isKeyWordFound:
            _print("Risky keyword found! Skipping TextAnalytics")
            return True

        # 2. Text Analytics ================================
        # Get sentiment analysis from TextAnalytics
        batched_text = [text]
        batched_result = self.ta_client.analyze_sentiment(batched_text, show_opinion_mining=True)
        result = batched_result[0]

        # Process the results
        if result.is_error:
            return False
        else:
            # Count number of negative sentences in the text result
            num_negative = 0
            for sentence in result.sentences:
                _print("Sentence negative confidence score is {}".format(sentence.confidence_scores['negative']))
                num_negative += 1 if isSentenceRisky(sentence) else 0
            # Record the riskiness
            _print("{} out of {} sentences are risky.".format(num_negative, len(result.sentences)))
            return num_negative >= neg_thresh * len(result.sentences)

if __name__ == '__main__':    
    rc = RiskClassifier(endpoint=ta_endpoint, key=ta_key)
    print("\nWelcome to Text Analytics Risk Classifier demo.")
    print("Type something and classifier will classify it.")
    print("Enter an empty input to end the demo.")
    for i in range(100):
        text = input("\nType something -> ")
        if len(text) == 0:
            break
        else:
            isRisky = rc.isRisky(text, verbose=True)
            print("Risky? ->", isRisky)
    print("Demo done")