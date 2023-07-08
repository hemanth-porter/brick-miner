import functools
import time
import numpy as np
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import openai
import string
import pandas as pd 
import os  
from tqdm import tqdm
import nltk 
from nltk.stem import WordNetLemmatizer  
from nltk.corpus import stopwords  
import string 
from tqdm import tqdm
import re 
import gensim 
from gensim import corpora, models   
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
from scipy.special import softmax
from config import model_path
from config import API_KEY
from streamlit_download import download_button

import streamlit as st

from constants import complaints_modelling_prompt,positive_points_modelling_prompt,change_format_prompt,theme_finder_prompt

openai.api_key = API_KEY


def retry(func):
        """
        
        retries executing the same function if it falils for 5 times after waiting for a second. 

        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            count = 0
            while count < 5:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"{func.__name__} failed with error: {str(e)}")
                    count += 1
                    time.sleep(2)
            raise Exception(f"{func.__name__} failed after 5 attempts.")
        
        return wrapper

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Elapsed time: {} seconds'.format(end_time - start_time))
        return result
    return wrapper




class ReviewAnalyser():


    def __init__(self, file_name,column_name):
        self.file_name = file_name
        self.column_name = column_name
        self.model_path = model_path
        self.data = None
        self.complaints_list = []
        self.positives_list = []
        self.file_upload_done = False

        



    def load_data(self):
        """
        Loads data from streamlit UI. Takes CSV Input 
        """
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            options = list(df.columns)
            col_name = st.selectbox('Select an option', options)

            suggestions = df[col_name].dropna()
            self.data = pd.DataFrame(suggestions)
            self.data.columns = ['review']
                
            
        else:
            st.write("No CSV file uploaded yet.")

        if st.button("Continue"):
            self.file_upload_done = True
            

    @staticmethod
    def preprocess(text):
        """
        
        """
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        words = word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # Join the words back into a string
        text = " ".join(words)
        return text

    def load_sentiment_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.autoconfig =  AutoConfig.from_pretrained(self.model_path)


    def get_sentiment_helper(self,text):

        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

    #     Print labels and scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        details_text = ''
        
        for i in range(scores.shape[0]):
            l = self.autoconfig.id2label[ranking[i]]
            s = scores[ranking[i]]
            details_text = details_text + f"{l}. {np.round(float(s), 4)} "
            
        if scores.argmax() == 1:
            return "Postive"
        elif scores.argmax() == 0:
            return "Negative"
    @timer
    def get_sentiment(self):

        self.data['sentiment'] = self.data.review.apply(lambda x: self.get_sentiment_helper(x) )
        
        
    def sentiment_splitter(self, labels = 2):

        data = self.data

        if labels ==2 :
            positive = data[data['sentiment'] == 'Postive'].copy()
            negative = data[data['sentiment'] == 'Negative'].copy()
            return positive, negative, None
        
        elif labels == 3:
            positive = data[data['sentiment'] == 'Postive'].copy()
            negative = data[data['sentiment'] == 'Negative'].copy()
            neutral = data[data['sentiment'] == 'Neutral'].copy()
            return positive, negative, neutral

    @retry
    def get_summary( self, df):

        list_of_reviews = df['review_concat']
        topic_name = df['exploded_topic']

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": f"Please provide concise, short and summarize a list of reviews accurately, without using any generalizations, to help readers easily understand the overall content of the reviews with respect to the topic name given. Keep it limited to 3 lines of paragraph"},
                {"role": "user", "content": f"List of reviews : {list_of_reviews}, Topic: {topic_name}"},
            ])

        message = response.choices[0]['message']['content']
        return message
    

    def ask_gpt(self,system_instruction, list_of_reviews):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": f"{system_instruction}"},
                {"role": "user", "content": f"{list_of_reviews}"},
            ])

        message = response.choices[0]['message']['content']
        return message
    
    def theme_finder(self, my_topics_list):
        theme_response = self.ask_gpt(theme_finder_prompt,my_topics_list)
        theme_repsose_dict = eval(theme_response)
        return theme_repsose_dict
    
    def get_theme_contri_metrics(self,df,theme_name,theme_topics_list):

        reviews_for_theme = df[df['exploded_topic'].isin(theme_topics_list)].copy()

        #no of unique reviews for the current theme
        unique_reviews_under_theme = reviews_for_theme['review'].nunique()

        # No of times Topic under this theme are talked ( Non - unique review)
        total_reviews_under_theme = len(reviews_for_theme['review'])

        #Therefore 472 times the topics are talked by 129 reviews.

        #Percentage contri of each topic under a theme
        topic_contri_df = pd.DataFrame(reviews_for_theme['exploded_topic'].value_counts()*100/len(reviews_for_theme))
        topic_contri_df = topic_contri_df.reset_index()
        topic_contri_df.columns = ['topic','topic_contri_for_theme']

        return theme_name, unique_reviews_under_theme,total_reviews_under_theme,topic_contri_df

    def themify(self,df):
        """
        df = analsyed_data_neg / analsyed_data_pos
        """
        df['exploded_topic'] = df['topic'].apply(lambda x: self.get_topic_explode(x))
        df_exploded = df.explode("exploded_topic")
        list_of_unique_topics = list(df_exploded['exploded_topic'].unique())

        theme_repsose_dict = self.theme_finder(list_of_unique_topics)

        themes_list = list(theme_repsose_dict.keys())

        theme_results = []
        for i in range(len(themes_list)):
            current_theme_results = self.get_theme_contri_metrics(df_exploded, themes_list[i], theme_repsose_dict[themes_list[i]])

            theme_results.append(current_theme_results)

        return theme_results
    
    def show_theme_results(self,individual_theme_results_list):
        theme_name, unique_reviews_under_theme,total_reviews_under_theme,topic_contri_df = individual_theme_results_list
        st.write("Theme Name: ",theme_name)
        st.write("Unique Reviews: ",unique_reviews_under_theme)
        st.write("Total Reviews: ",total_reviews_under_theme)
        st.write(topic_contri_df)
        

        

    @retry
    def topic_modeling_gpt(self, list_of_reviews,what):
        if what == 'complaints':
            return self.ask_gpt(complaints_modelling_prompt.format(self.complaints_list), list_of_reviews)
        elif what == 'positives':
            return self.ask_gpt(positive_points_modelling_prompt.format(self.positives_list), list_of_reviews)
    

    @retry
    def handle_failed_gpt_modeling(self,failed_str,actual_review):
        """
        Return if the len of review < 25 ( Tunable )
        Else, Tries running formatter 
        """

        if len(actual_review) < 25:
            return actual_review
        
        else:
            return self.ask_gpt(change_format_prompt,failed_str)
        

    def add_point_to_df(self,df,indx,return_string):
        df.iloc[indx,df.columns.get_loc('topic')] = return_string
        
    def add_new_points_to_list(self, return_string, what):    
        """
        return_string = 'returned complaints/positive points from gpt'
        what = 'complaints/positives'
        """
        return_list = eval(return_string)

        if what == 'complaints':
            for each_new_complaint in return_list:
                if each_new_complaint not in self.complaints_list:
                    self.complaints_list.append(each_new_complaint)

        elif what == 'positives':
            for each_positive_point in return_list:
                if each_positive_point not in self.positives_list:
                    self.positives_list.append(each_positive_point)


    def add_actual_review_to_df_and_list(self, df,indx):

        # df = self.data

        actual_review = f"['{df['review'].iloc[indx]}']"
        df.iloc[indx,df.columns.get_loc('topic')] = actual_review
        self.complaints_list.append(actual_review)

    @timer
    def get_analysis(self, df, what  ):

        df['error'] = np.nan
        df['second_error'] = np.nan

        df['topic'] = np.nan
        self.failed_indx = []
        for indx in range(len(df['review'])):
        
            each_review = df['review'].iloc[indx]
            return_string = self.topic_modeling_gpt(each_review,what)
            
            try:
                self.add_new_points_to_list(return_string,what)
                self.add_point_to_df(df,indx,return_string)

            except Exception as e:

                print(e)
                print(f"Faied for {indx}: {return_string}")
                self.failed_indx.append(indx)
                df.iloc[indx,df.columns.get_loc('error')] = e
            
                corrected_str = self.handle_failed_gpt_modeling(return_string,each_review)
                try:
                    #checks if the corrected string can be added to list and df"
                    self.add_new_points_to_list(corrected_str,what)
                    self.add_point_to_df(df,indx,corrected_str)
                except Exception as e2:
                    print(f"Failed for second time {return_string}, {corrected_str}")
                    df.iloc[indx,df.columns.get_loc('second_error')] = e2
                    
                    print()

        return df
    
    def download_streamlit_csv(self,df):
        df = df.reset_index(drop = True).copy()
        st.write(df)
        st.download_button("Download as CSV",
                           df.to_csv(index = Falsed),
                           mime = 'text/csv'
                           )
        
        

    def show_metrics(self,total,pos,neg):

        pos_percent = round(pos*100/total,2)
        neg_percent = round(neg*100/total,2)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total non-null reviews", str(total))
        col2.metric("Positive reviews", str(pos)+f" ( {pos_percent}% )")
        col3.metric("Negative reviews", str(neg)+f" ( {neg_percent}% )")
        
        
    def get_topic_priority_and_percentage(self,df):

        df['exploded_topic'] = df['topic'].apply(lambda x: self.get_topic_explode(x))
        df = df.explode("exploded_topic").copy()
        df_grouped = df.groupby(['exploded_topic'],as_index = False).agg({"sentiment":'count',
                                                                        "review":'unique'
                                                                        })

        df_grouped.sort_values(by = 'sentiment',ascending = False,inplace = True)
        

        df_grouped['topic_percentage'] = df_grouped['sentiment'].apply(lambda x: x*100/sum(df_grouped['sentiment']))
        df_grouped['review_concat'] = df_grouped['review'].apply(lambda x: ",".join(x))

        df_grouped['summaried_reviews'] = df_grouped.apply(lambda x: self.get_summary(x), axis = 1)

        return_df = df_grouped[['exploded_topic','sentiment', 'topic_percentage','review','summaried_reviews']]

        return_df.columns = ['topics','n_reviews_for_this_topic','topic_percentage','raw_reviews','summaried_reviews']

        return return_df


    def get_topic_explode(self, topic_list):
        try:
            return eval(topic_list)
        except:
            return []


if __name__ == "__main__" :

    app = ReviewAnalyser('TEMP','TEMP')

    app.load_data()
    print("LOADED DATA")

    

    if app.file_upload_done:
        app.load_sentiment_model()
        print("LOADED SENTIMENT MODEL")

        app.get_sentiment()
        print("GOT SENTIMENT")

        positive, negative, neutral = app.sentiment_splitter()        
        print("SPLIT DATA DONE..")

        download_button_str = download_button(negative,'negative_data.csv','Click to donwload negative')
        st.markdown(download_button_str, unsafe_allow_html=True)

        app.show_metrics(len(app.data),len(positive),len(negative))

        analsyed_data_neg = app.get_analysis(negative.iloc[:10], "complaints")
        print("ANALYSED COMPLAINTS DATA")
    
        # analsyed_data_neg.to_csv("output/analsyed_data_neg.csv",index = False)

        review_summaried_neg = app.get_topic_priority_and_percentage(analsyed_data_neg)
        

        results = app.themify(analsyed_data_neg)

        for each_result in results:
            app.show_theme_results(each_result)
        
        st.stop()