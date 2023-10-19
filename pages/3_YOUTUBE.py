import time
import streamlit as st
from pytube import YouTube

from runtime import model
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter

st.title("SuperChat with Youtube transcripts")
st.markdown(
    "**Chat with Claude v2 on Bedrock (100k context)")

if 'doc_youtube' not in st.session_state:
    st.session_state['doc_youtube'] = ""

url = ''
if st.experimental_get_query_params():
    url = st.experimental_get_query_params()['url'][0]

input_url = st.text_input("Enter a URL:", value=url)

if "messages" not in st.session_state:
    st.session_state.messages = []

if input_url and st.session_state['doc_youtube'] == "":
    with st.spinner('Processing'):
        yt = YouTube(input_url)
        stream = yt.streams.first()

        details = f"""
        title: {yt.title},\n\n
        author: {yt.author},\n\n
        
        * length: {yt.length},\n\n
        * views: {yt.views},\n\n
        ---
        description: {yt.description},\n\n
        ---
        keywords: {yt.keywords},\n\n
        """

        with st.chat_message("assistant"):
            st.info(details)
            st.experimental_set_query_params = {'url': input_url}

        transcript = YouTubeTranscriptApi.get_transcript(yt.video_id)
        formatter = SRTFormatter()
        text = formatter.format_transcript(transcript)

        with st.chat_message("assistant"):
            st.info(f"Load transcript len: {len(text)}")

        prompt_template = f"""
        I get youtube transcripts and base info for you.

        <info>
        {details}
        </info>

        Here is the transcript in SRT format:
        <transcript>
        {text}
        </transcript>

        Thus, the format of your overall response should look like example what's shown between the <example></example> tags.  
        For generating a script that dynamically adapts to the length of the input text,The aim would be to maintain the integrity of the essential points while condensing information if the text is too long.
        Make sure to follow the formatting and spacing exactly. 

        <example>
        # title
        ## subtitle

        ### Summary:  
        [Generate a summarized and storytelling narrative from the following text.]
        ---
        ### Scenes (with timecodes):
        [Divide the into three acts, 5-10 scenes. ]
        [including, if possible, descriptions, quotes and characters]
        
        ### Chapters (by scenes split by timestamps for YouTube in detail video chapters)
        00:00 Introduction
        01:30 Chapter 1: Basics

        ---
        ### Analysis (with timecodes):
        (1) Identify the main themes and problems discussed.
        (2) List interesting theses and quotes.
        (3) Identify the main characters.
        (4) Suggest tags for linking with articles.
        (5) Sentiment Analysis. Is the sentiment expressed in the text positive, negative, or neutral? Please provide evidence from the text to support your assessment. Additionally, on a scale of -1 to 1, where -1 is extremely negative, 0 is neutral, and 1 is extremely positive, rate the sentiment of the text.
        (6) Political Orientation Analysis. 
        (7) Fake news detection or manipulation, critical thinking.

        </example>

        Answer the question immediately without preamble.
        Result in Markdown format.
        Answer in 8000 words or less.

        """

        news_summarise = model.predict(input=prompt_template)
        st.session_state['doc_youtube'] = news_summarise
        st.session_state.messages.append({"role": "assistant", "content": news_summarise})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for chat input if it doesn't already exist

prompt_disabled = (st.session_state['doc_youtube'] == "")

if prompt := st.chat_input("What is up?", disabled=prompt_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        processed_prompt = prompt

        result = model.predict(input=prompt)

        for chunk in result:
            full_response += chunk
            time.sleep(0.01)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
