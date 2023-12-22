import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import OpenAIChat
from langchain.retrievers import AmazonKendraRetriever
from langchain.llms.bedrock import Bedrock
import boto3



def test():
    a = 1+1
