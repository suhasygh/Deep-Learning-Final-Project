from dotenv import load_dotenv
import os
from openai import OpenAI
from llama_index.agent import OpenAIAssistantAgent
import base64
import streamlit as st
from fpdf import FPDF

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def initialize_assistant_agent(documents):
    """
    Initializes the OpenAI Assistant Agent with LlamaIndex for document-based knowledge.
    
    :param documents: List of document filenames to be used by the agent.
    :return: Initialized OpenAIAssistantAgent object.
    """

    bot_instructions = """
    As Mock Interview Bot, my primary role is to conduct unbiased and professional job interviews, focusing solely on assessing candidates' skills and suitability for a role. 
    I engage users with one question at a time, tailoring each subsequent question based on their previous responses to ensure a relevant and personalized interview experience. 
    The interview process is deliberately concise, limited to a maximum of four questions to ensure depth and focus. After the final question has been asked and a response received, I express gratitude to the user for their participation. 
    Following the conclusion of the interview, I provide a concise, informational report analyzing the user's overall performance. 
    This analysis covers aspects such as response content, nonverbal cues, and presentation skills, culminating in a downloadable PDF report. 
    This document offers users the opportunity to review and reflect on their performance, aiding in their interview preparation and professional development efforts. 
    This approach ensures a targeted and effective evaluation, supporting the user's growth and readiness for their career journey.
    """

    agent = OpenAIAssistantAgent.from_new(
        name="S.A.R.A",
        instructions=bot_instructions,
        openai_tools=[{"type": "retrieval"}],
        instructions_prefix="Please focus on assessing the candidate's skills and suitability for the role.",
        files=documents,
        verbose=True,
    )
    return agent

def get_answer(agent, user_input):
    """
    Generates an answer from the OpenAI Assistant Agent based on the user's input.
    
    :param agent: The initialized OpenAIAssistantAgent object.
    :param user_input: User's input as a string.
    :return: Agent's response as a string.
    """
    response = agent.chat(user_input)
    return str(response)

def speech_to_text(audio_data):
    """
    Converts speech to text using OpenAI's API.
    """
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        response_format="text",
        file=open(audio_data, "rb")
    )
    return transcript

def text_to_speech(input_text):
    """
    Converts text to speech using OpenAI's API.
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

def generate_pdf_report(content, filename="interview_feedback.pdf"):
    """
    Generates a PDF report with the given content.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(filename)
    return filename
