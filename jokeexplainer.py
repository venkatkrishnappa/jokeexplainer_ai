import streamlit as st
from openai import OpenAI
import os
#from azure.ai.inference import ChatCompletionsClient
#from azure.ai.inference.models import SystemMessage, UserMessage
#from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()  # take environment variables
import pandas as pd
import requests  # Assuming you'll make API calls to the models

GIT_PAT=os.getenv("GIT_PAT")

#function to call openai model
def invoke_openai_llm(joke,model_name):
    endpoint = "https://models.inference.ai.azure.com"
    token=GIT_PAT
    client = OpenAI(
        base_url=endpoint,
        api_key=token,
        )
    response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who will explain the nuances of the joke shared by user",
                },
                {
                    "role": "user",
                    "content": joke,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
    
    explanation=response.choices[0].message.content
    return explanation


# Function to call the models with the joke
def get_explanation_from_model(joke, model):
    # Replace this URL with the actual endpoint of your models
    if model == 'gpt-4o-mini':
        response=invoke_openai_llm(joke,model)
    else:
        response="Not implemented"

    return response

# Setting up the Streamlit application
st.title("Joke Explainer and Model Comparer")

# Input box for the joke
joke_input = st.text_input("Enter your joke:")

# Submit button
if st.button("Submit"):
    if joke_input:
        # Get explanations from both models
        gpt_explanation = get_explanation_from_model(joke_input, "gpt-4o-mini")
        phi_explanation = get_explanation_from_model(joke_input, "phi-4-mini-instruct")

        # Create a DataFrame for displaying the results
        results_df = pd.DataFrame({
            "Model": ["GPT-4o-mini", "Phi-4-mini-instruct"],
            "Explanation": [gpt_explanation, phi_explanation]
        })

        # Display results in tabular form
        st.table(results_df)
    else:
        st.warning("Please enter a joke to submit.")