from fastapi import FastAPI, Form, Request, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import openai
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
nltk.download('punkt')  # punkt tokenizer
from fastapi.middleware.cors import CORSMiddleware
import requests
from requests.auth import HTTPBasicAuth
import mysql.connector
import uuid
import time
from flask import Flask
from flask_cors import CORS
import psutil
import time
import re
import ast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def pre_process(message):
    if message.startswith("Bot:") and "Bot:" in message and "Customer:" in message:
        customer_messages = re.findall(r'Customer:\s(.*?)(?=\sBot:|\Z)', message)
        val = customer_messages[0]
        print("Preprocess",val)
        return(val)
    else:
        return(message)

def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    print("inside connect_unix_socket")
    db_user = "root"  
    db_pass = "genai-storage"  
    db_name = "genai_sql" 
    unix_socket_path = "/cloudsql/tradeday:us-central1:genai-storage" 

    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="mysql+pymysql",
            username=db_user,
            password=db_pass,
            database=db_name,
            query={"unix_socket": unix_socket_path},
        ),
    )
    print("pool: ",pool)
    return pool

def execQueryRaw(query, params=None, fetch_data=True):
    engine = connect_unix_socket()
    stmt = sqlalchemy.text(query) 
    try:
        with engine.connect() as conn:
            if params:
                result = conn.execute(stmt, params)
                print("Inside params", result)
            else:
                result = conn.execute(stmt)
                print("Inside else", result)
            if fetch_data:
                fetched_data = result.fetchall()
                print("Fetched Data: ", fetched_data)
                columns = ["convo", "subject", "type", "tokenized_convo", "tokenized_subject", "convo_vector", "subject_vector", "tokenized_filtered_convo", "synonyms_tokenized_convo", "synonyms_tokenized_convo_vector", "links"]
                data = pd.DataFrame(fetched_data, columns=columns)
                print("Inside fetch_data", data)
                return data
            else:
                return None 
    except Exception as e:
        print("Encountered exception:", str(e))
        return "Exception"
    
def execQueryRawTicket(query, params=None, fetch_data=True):
    engine = connect_unix_socket()
    stmt = sqlalchemy.text(query) 
    try:
        with engine.connect() as conn:
            if params:
                result = conn.execute(stmt, params)
                print("Inside params", result)
            else:
                result = conn.execute(stmt)
                print("Inside else", result)
            if fetch_data:
                fetched_data = result.fetchall()
                print("Fetched Data: ", fetched_data)
                columns = ["convo", "subject", "type", "convo_vector", "subject_vector"]
                ticket_data = pd.DataFrame(fetched_data, columns=columns)
                print("Inside fetch_data", ticket_data)
                return ticket_data
            else:
                return None 
    except Exception as e:
        print("Encountered exception:", str(e))
        return "Exception"    

word2vec_model_convo = gensim.models.Word2Vec.load('word2vec_model_convo_latest')
word2vec_model_subject = gensim.models.Word2Vec.load('word2vec_model_subject_latest')

def average_word_vectors(words, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    for word in words:
        if word in model.wv.key_to_index:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])
    if nwords > 0:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def string_to_array(string):
    string = re.sub(r'\s+', ',', string.strip())  
    string = re.sub(r'(?<=\d)\s+(?=\d)', ',', string)  
    string = string.replace('[,', '[').replace(',]', ']')  
    try:
        return np.array(eval(string))
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string to array: {e}")
        return np.array([])
 
def find_similar_conversations(convo, subject, data, ticket_data):
    new_email_convo_tokenized = word_tokenize(convo.lower())
    new_email_convo_vector = average_word_vectors(new_email_convo_tokenized, word2vec_model_convo, 100)

    # Similar FAQS:
    data['convo_vector'] = data['convo_vector'].apply(string_to_array)
    print(data['convo_vector'].head())
    print("pass string to array")

    data['similarity_score_convo'] = data['convo_vector'].apply(lambda vec: cosine_similarity([vec], [new_email_convo_vector])[0][0])

    data = data.sort_values(by='similarity_score_convo', ascending=False)
    
    similar_emails = data.head(3)
    print(similar_emails)

    similar_conversations = []
    similar_links = []
    similarity_scores = []
    
    for index, row in similar_emails.iterrows():
        print("inside 1st for loop")
        if row['similarity_score_convo'] > 0.87:
            similar_conversations.append(f"{row['convo']} with a similarity score of {row['similarity_score_convo']:.2f}")
            similar_links.append(row['links'])
            similarity_scores.append(row['similarity_score_convo'])

    if similar_conversations:
        print("Found similar faq")
        combined_conversations = "Examples of previous similar conversations: " + ', '.join(similar_conversations)
        combined_links = "Examples of previous similar links: " + ', '.join(similar_links)
        print("Similar Convo:  ", combined_conversations, " and combined links: ", combined_links)
        return combined_conversations, similarity_scores, combined_links

    # Similar FAQS With synonymns:
    if not similar_conversations:
        similar_conversations = []
        similar_links = []
        similarity_scores = []
        print("Inside Similar FAQS With synonymns")
        data['synonyms_tokenized_convo_vector'] = data['synonyms_tokenized_convo_vector'].apply(string_to_array)

        print(data['synonyms_tokenized_convo_vector'].head(5))
        print("pass synonyms_tokenized_convo_vector string to array")
    
        data['similarity_score_syno_convo'] = data['synonyms_tokenized_convo_vector'].apply(lambda vec: cosine_similarity([vec], [new_email_convo_vector])[0][0])

        data = data.sort_values(by='similarity_score_syno_convo', ascending=False)
        
        similar_emails = data.head(3)
        print(similar_emails)

        for index, row in similar_emails.iterrows():
            print("inside 2nd for loop")
            if row['similarity_score_syno_convo'] > 0.87:
                print("Score: ",row['similarity_score_syno_convo'])
                similar_conversations.append(f"{row['convo']} with a similarity score of {row['similarity_score_syno_convo']:.2f}")
                similar_links.append(row['links'])
                similarity_scores.append(row['similarity_score_syno_convo'])

    if similar_conversations:
        print("Found similar faq with synonymns")
        combined_conversations = "Examples of previous similar conversations: " + ', '.join(similar_conversations)
        combined_links = "Examples of previous similar links: " + ', '.join(similar_links)
        print("Similar Convo:  ", combined_conversations, " and combined links: ", combined_links)
        return combined_conversations, similarity_scores, combined_links
    
    # Similar Previous Tickets:
    if not similar_conversations:
        similar_conversations = []
        similar_links = []
        similarity_scores = []
        print("Inside similar Previous Tickets")
        ticket_data['convo_vector'] = ticket_data['convo_vector'].apply(string_to_array)

        print("Ticket Convo Vector: ", ticket_data['convo_vector'].head(5))
    
        ticket_data['similarity_score'] = ticket_data['convo_vector'].apply(lambda vec: cosine_similarity([vec], [new_email_convo_vector])[0][0])

        ticket_data = ticket_data.sort_values(by='similarity_score', ascending=False)
        
        similar_emails = ticket_data.head(3)
        print(similar_emails)

        for index, row in similar_emails.iterrows():
            print("inside 3rd for loop")
            if row['similarity_score'] > 0.93:
                print("Score: ",row['similarity_score'])
                similar_conversations.append(f"{row['convo']} with a similarity score of {row['similarity_score']:.2f}")
                similarity_scores.append(row['similarity_score'])

    if similar_conversations:
        print("Found similar ticket")
        combined_conversations = "Examples of previous similar conversations: " + ', '.join(similar_conversations)
        print("Similar Convo:  ", combined_conversations)
        return combined_conversations, similarity_scores, []
    


    print("Don't found similar conversation")
    return "Don't found similar conversation",[],[]

@app.post("/test_faqs")
async def test_faqs(request: Request):
    request_data = await request.json()
    print("Input: ",request_data)
    new_email_id = request_data["new_email_id"]
    new_email_convo = request_data["new_email_convo"]
    new_email_subject= request_data["new_email_subject"]
    print("new_email_subject: ", new_email_subject)
    print("Type of new_email_subject: ", type(new_email_subject))
    new_email_convo = pre_process(new_email_convo)

    print("connecting to database")
    sql1= "SELECT convo, subject, type, tokenized_convo,tokenized_subject,	convo_vector, subject_vector, tokenized_filtered_convo, synonyms_tokenized_convo, synonyms_tokenized_convo_vector,links FROM vectorize_newfaqs "
    data = execQueryRaw(sql1)
    print("Faqs fetched: ", data.head(5))
    sql2= "SELECT convo, subject, type, convo_vector, subject_vector FROM vectorize_data WHERE type = 'ticket'"
    ticket_data = execQueryRawTicket(sql2)
    print("Tickets fetched: ", ticket_data.head(5))
    

    combined_conversations, similarity_score, combined_links = find_similar_conversations(new_email_convo, new_email_subject, data, ticket_data)

    print("Similarity Score: ", similarity_score)

    if combined_conversations == "Don't found similar conversation":
        return "I don't know the answer."
    else:
        openai.api_key = ""
        max_tokens = 5600
        request_body = {
            "model": "gpt-4",
            "temperature" : 0.1,
            "messages": [
                {
                    "role": "system", "content": "You are an expert agent with ten years of experience of Tradeday (It is a trading education, and trader evaluation company.  It was established in 2020 and is registered in Illinois, USA. TradeDay also has a presence in the UK. It is owned and managed by co-founders James Thorpe and Steve Miley, who have a combined 60 years of international trading and futures industry experience.) Your role as a seasoned customer support representative is to provide customer support. Only give factual answers and don't be creative."
                    },
                {
                    "role": "user","content": ("Clear your memory . Forget everything that you have answered and start afresh. Generate a reply for the following question being asked by user:"+ str(new_email_convo)+ 
                                            "Some examples of similar conversation with there similarity score value are as follows- "+ str(combined_conversations[:max_tokens])+
                                            "You can use the similar examples to fetch information to formulate the answer. Make sure the context is limited to information passed. Also make sure that the response is precise , realistic while maintaining complete context."+ 
                                            "BE ACCURATE, ONLY ANSWER IF YOU ARE SURE. DO NOT HALLUCINATE. IF UNSURE SAY I DON'T KNOW. DO NOT CREATE ANSWER ON YOUR OWN. Please mantain the same tone from the previous answers. Your response should be not greater than 50 words. Restrict your response to the information that have been shared on the prompt"
                                            )
                    },
            ],
        }
        response = openai.ChatCompletion.create(**request_body)
        reply_text = response.choices[0].message["content"]
        print(reply_text)
        response = {
            "note_response": reply_text,
            "combined conversations" : combined_conversations[:max_tokens],
            "links": combined_links


        }
        print("Response of Openai", response)

        return response 

        