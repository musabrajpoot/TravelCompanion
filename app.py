# Import necessary libraries
import streamlit as st
from transformers import pipeline

# Initialize pipelines with explicit pad_token_id and tokenizer
qa_pipeline = pipeline(
    "question-answering", 
    model="deepset/roberta-base-squad2", 
    tokenizer="deepset/roberta-base-squad2"
)

recommend_pipeline = pipeline(
    "text-generation", 
    model="gpt2", 
    tokenizer="gpt2", 
    pad_token_id=50256
)

translate_pipeline = pipeline(
    "translation_en_to_es", 
    model="Helsinki-NLP/opus-mt-en-es", 
    tokenizer="Helsinki-NLP/opus-mt-en-es"
)

# Define chatbot functions
def answer_faq(question, context):
    return qa_pipeline({"question": question, "context": context})['answer']

def generate_recommendation(prompt):
    return recommend_pipeline(prompt, max_length=50, num_return_sequences=1, pad_token_id=50256)[0]['generated_text']

def translate_to_spanish(phrase):
    return translate_pipeline(phrase)[0]['translation_text']

# Streamlit App
st.title("Travel Companion Chatbot")
st.write("Your personal travel assistant for FAQs, recommendations, and translations!")

# FAQ Section
st.header("FAQ Assistance")
faq_question = st.text_input("Enter your question:")
faq_context = st.text_area("Provide the context (e.g., travel rules, destinations):")
if st.button("Get Answer"):
    if faq_question and faq_context:
        answer = answer_faq(faq_question, faq_context)
        st.success(f"Answer: {answer}")
    else:
        st.error("Please provide both a question and context.")

# Recommendation Section
st.header("Destination Recommendations")
recommend_prompt = st.text_input("Enter a travel theme or prompt (e.g., 'beach destinations'):")
if st.button("Get Recommendation"):
    if recommend_prompt:
        recommendation = generate_recommendation(recommend_prompt)
        st.success(f"Recommendation: {recommendation}")
    else:
        st.error("Please provide a prompt for recommendations.")

# Translation Section
st.header("Translate to Spanish")
phrase_to_translate = st.text_input("Enter a phrase in English:")
if st.button("Translate"):
    if phrase_to_translate:
        translation = translate_to_spanish(phrase_to_translate)
        st.success(f"Translation: {translation}")
    else:
        st.error("Please provide a phrase to translate.")

# Footer
st.markdown("---")
st.caption("Powered by Hugging Face Transformers and Streamlit.")
