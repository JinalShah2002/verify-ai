"""

@author: Jinal Shah

This script will be my preprocessing 
pipeline to take the dataset from 
raw to training.

"""
# Importing libraries
import sqlalchemy
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import pandas as pd
from torchtext.data import get_tokenizer
import language_tool_python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Adding the credentials
sys.path.append('../')
from credentials import credentials

# Getting tqdm for pandas 
tqdm.pandas()

# Class for preprocessing
class Preprocessing():
    # Constructor
    def __init__(self,data_df:pd.DataFrame):
        self.data = data_df
        self.stop_words = stopwords.words('english')
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # tokenizer for words only
        self.overall_tokenizer = get_tokenizer('spacy',language='en_core_web_sm')
        self.grammar_checker = language_tool_python.LanguageTool('en-US',config={'cacheSize': 5000,'maxCheckThreads':20})
        self.detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
        self.detector = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.emotion_detector = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.emotions = ['anger','disgust','fear','joy','neutral','sadness','surprise']

        print('Tokenizing the essays into only words...')
        self.data['tokenized_essay_words'] = self.data['essay'].progress_apply(self.tokenize_essay_words)
        print()
        print('Tokenizing the essays into overall tokens with words and punctuation')
        self.data['tokenized_overall'] = self.data['essay'].progress_apply(self.tokenize_overall)
        print()
    
    # Function to tokenize each essay into words only
    def tokenize_essay_words(self,essay:str) -> list:
        return self.tokenizer.tokenize(essay)
    
    # Function to tokenize each essay into overall tokens
    def tokenize_overall(self,essay:str) -> list:
        return self.overall_tokenizer(essay)
    
    # Getting the count of stop words in an essay
    def get_stop_word_count(self,text:str) -> int:
        count = 0
        for word in text:
            if word in self.stop_words:
                count += 1
        return count
    
    # Getting the count of the unique words in an essay
    def get_unique_words(self,essay:list) -> int:
        return len(set(essay))
    
    # Getting the counts of each punctuation (?, !, ;, :)
    def count_punc(self, essay:list) -> tuple[int,int,int,int]:
        count_q = 0
        count_ex = 0
        count_semi = 0
        count_col = 0

        # Iterating through the tokenized essay
        for token in essay:
            if token == "?":
                count_q += 1
            elif token == "!":
                count_ex += 1
            elif token == ";":
                count_semi += 1
            elif token == ":":
                count_col += 1
        
        return count_q, count_ex,count_semi, count_col
    
    # A function to get the number of grammatical errors
    def get_grammar_error_count(self,essay:str) -> int:
        errors = self.grammar_checker.check(essay)
        return len(errors)
    
    # A function to get the detection from OpenAI's GPT-2 Detector
    def get_detect_pred(self,text:str) -> int:
        # Tokenizing the input essay
        inputs = self.detector_tokenizer(text,return_tensors='pt',truncation=True)

        # Getting the logits
        with torch.no_grad():
            logits = self.detector(**inputs).logits

        # Doing 1 - max logit because the model has "Real" = class 1 and "Fake" = class 0
        # My labels are the opposite, 1 = LLM Written and 0 = student written.
        # If a logit = 0 = Fake, 1-0 = 1 = LLM Written
        # If a logit = 1 = Real, 1-1 = 0 = student written
        predicted_class = 1 - logits.argmax().item()
        return predicted_class
    
    # A function to get the prediction from the Emotion Detector
    def emotion_detector_pred(self,essay:str) -> tuple[int,int,int,int]:
        # Tokenizing the input essay
        inputs = self.emotion_tokenizer(essay,return_tensors='pt',truncation=True)

        # Getting the logits
        with torch.no_grad():
            logits = self.emotion_detector(**inputs).logits

        # Getting the predicted emotion
        predicted_emotion = self.emotions[logits.argmax().item()]
        if predicted_emotion == 'anger':
            return 1,0,0,0
        elif predicted_emotion == 'surprise':
            return 0,1,0,0
        elif predicted_emotion == 'sadness':
            return 0,0,1,0
        elif predicted_emotion == 'fear':
            return 0,0,0,1
        else:
            return 0,0,0,0
    
    # A function to knit all preprocessing together
    def preprocessing(self) -> pd.DataFrame:
        # Getting the stop word count and the stop word ratio
        print('Adding the stop word features...')
        self.data['stop_word_count'] = self.data['tokenized_essay_words'].progress_apply(self.get_stop_word_count)
        self.data['stop_word_ratio'] = self.data['stop_word_count'] / self.data['word_count']
        print()
        
        # Getting the unique word counts and the unique word ratio
        print('Adding the unique word features...')
        self.data['unique_word_count'] = self.data['tokenized_essay_words'].progress_apply(self.get_unique_words)
        self.data['unique_word_ratio'] = self.data['unique_word_count'] / self.data['word_count']
        print()

        # Adding the punctuation features
        print('Adding the punctuation features...')
        punc_counts = self.data['tokenized_overall'].progress_apply(self.count_punc)
        self.data['count_question'] = [row[0] for row in punc_counts]
        self.data['count_exclamation'] = [row[1] for row in punc_counts]
        self.data['count_semi'] = [row[2] for row in punc_counts]
        self.data['count_colon'] = [row[3] for row in punc_counts]
        print()

        # Getting the number of grammatical errors
        print('Getting grammar error counts...')
        self.data['grammar_errors'] = self.data['essay'].progress_apply(self.get_grammar_error_count)
        print()

        # Adding the detector model prediction
        print('Getting Detector Prediction...')
        self.data['detector_pred'] = self.data['essay'].progress_apply(self.get_detect_pred)
        print()

        # Getting the emotion model prediction
        print('Getting Emotion Prediction...')
        emotion_rows = self.data['essay'].progress_apply(self.emotion_detector_pred)
        self.data['anger_pred'] = [row[0] for row in emotion_rows]
        self.data['surprise_pred'] = [row[1] for row in emotion_rows]
        self.data['sadness_pred'] = [row[2] for row in emotion_rows]
        self.data['fear_pred'] = [row[3] for row in emotion_rows]
        print()

        # Dropping the tokenized parts
        self.data.drop(['tokenized_essay_words','tokenized_overall'],axis=1,inplace=True)

        # returning the preprocessed dataframe
        return self.data
    
# Main method
if __name__ == '__main__':
    # Creating the database engine 
    connector_string = f'mysql+mysqlconnector://{credentials["user"]}:{credentials["password"]}@{credentials["host"]}/AuthenticAI'
    db_engine = sqlalchemy.create_engine(connector_string,echo=True)

    # Opening up a connection
    with db_engine.connect() as connection:
        data = pd.DataFrame([i for i in connection.execute(sqlalchemy.text('select * from essays;'))])
        data.drop(['prompt'],axis=1,inplace=True)

        # Running data through pre-processing
        preprocessor = Preprocessing(data)
        preprocessed_data = preprocessor.preprocessing()

        # Saving the csv file
        preprocessed_data.to_csv('../preprocessed_data.csv',index=False)

    # Closing the enginer
    db_engine.dispose()