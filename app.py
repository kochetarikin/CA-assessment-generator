import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_document_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQgenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

load_dotenv()

# Load JSON templates for different question types
with open('response_mcqs.json', 'r') as file:
    MCQ_RESPONSE_JSON = json.load(file)

with open('response_true_false.json', 'r') as file:
    TRUEFALSE_RESPONSE_JSON = json.load(file)

with open('response_descriptive.json', 'r') as file:
    DESCRIPTIVE_RESPONSE_JSON = json.load(file)

st.title("Question Generator")

question_type = st.radio("Select question type:", ["Multiple Choice", "True/False", "Descriptive"])

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    question_count = st.number_input("Number of questions", min_value=3, max_value=50)
    subject = st.text_input("Insert subject", max_chars=50)
    tone = st.text_input("Complexity level of questions", max_chars=50, placeholder="Simple")
    button = st.form_submit_button("Generate Questions")

    if button and uploaded_file is not None and question_count and subject and tone:
        with st.spinner("Generating questions..."):
            try:
                text = read_file(uploaded_file)
                
                if question_type == "Multiple Choice":
                    response_json = MCQ_RESPONSE_JSON
                elif question_type == "True/False":
                    response_json = TRUEFALSE_RESPONSE_JSON
                elif question_type == "Descriptive":
                    response_json = DESCRIPTIVE_RESPONSE_JSON
                
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain({
                        "text": text,
                        "number": question_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(response_json),
                        "question_type": question_type
                    })
            except Exception as e:
                logging.error(f"Error in generate_evaluate_chain: {str(e)}")
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(f"An error occurred: {str(e)}")
            else:
                logging.info(f"Total tokens:{cb.total_tokens}")
                logging.info(f"Prompt tokens:{cb.prompt_tokens}")
                logging.info(f"Completion tokens:{cb.completion_tokens}")
                logging.info(f"Total cost:{cb.total_cost}")

                if isinstance(response, dict):
                    quiz = response.get('quiz', None)
                    if quiz is not None:
                        logging.info(f"Received quiz: {quiz}")
                        document_data = get_document_data(quiz, question_type)
                        if document_data:
                            st.markdown(f"## Generated {question_type} Questions")
                            st.markdown(document_data)
                        else:
                            st.error("Failed to generate questions. The result was empty.")
                            logging.error("get_document_data returned empty string")
                    else:
                        st.error("Failed to generate questions. The 'quiz' key is missing from the response.")
                        logging.error(f"Quiz key missing from response: {response}")
                else:
                    st.error("Unexpected response format. Please try again.")
                    logging.error(f"Unexpected response format: {response}")