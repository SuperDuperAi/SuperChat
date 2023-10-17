import time
import boto3
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from botocore import config


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(text)
    return chunks


with open("styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

option = st.sidebar.radio("Choose an option:", ['Chat', 'Chat with PDF'])

st.title("SuperChat Ai with PDF")
st.markdown(
    "**Chat with Claude v2 on Bedrock (100k context). Get started by uploading a PDF!**")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    config=config.Config(
        connect_timeout=1000,
        read_timeout=3000
    )
)

# add to sidebar inputs max_tokens_to_sample
st.sidebar.subheader('Model parameters')
max_tokens_to_sample = st.sidebar.slider('tokens to answer', 256, 8000, 2048)


@st.cache_resource
def load_llm():
    llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
    llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": max_tokens_to_sample}

    DEFAULT_TEMPLATE = """{history}\n\nHuman: {input}\n\nAssistant:"""
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=DEFAULT_TEMPLATE
    )

    model = ConversationChain(
        prompt=prompt,
        llm=llm,
        # verbose=True,
        memory=ConversationBufferMemory(
            human_prefix="\n\nHuman: ",
            ai_prefix="\n\nAssistant:"
        )
    )

    return model


model = load_llm()

pdf_docs = None
if 'doc' not in st.session_state:
    st.session_state['doc'] = ""

instruct_value = ""
instruct_text = ""

if option == "Chat with PDF":
    with st.sidebar:
        st.subheader('Your PDF documents')
        chunk_size = st.sidebar.slider('chunk_size', 0, 10000, 1000)
        pdf_chunks_limit = st.sidebar.slider('pdf_chunks_limit', 0, 95000, 90000)

        pdf_docs = st.file_uploader(
            "Upload your pdfs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])

        if st.button('Process'):
            with st.spinner('Processing'):
                text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=chunk_size,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                data = ""
                for chunk in chunks:
                    if len(data) > 90000:
                        st.warning('PDFs is big, only first >100k characters will be used')
                        break
                    data += chunk

                st.success('PDFs extracted to data string')
                st.session_state['doc'] = data
                st.success(f'Text chunks generated, total words: {len(st.session_state["doc"])}')

    instruct_value = """
    Then, I'll ask you to create an extensive long-read article suitable for blog publication based on the information. 
    Please adhere to the following sections and guidelines in your response:

    Literary Analysis:
    a. Main Themes and Challenges: Identify and discuss the overarching themes and problems.
    b. Engaging Theses and Quotations: List interesting theses and quotes.
    c. Principal Characters: Identify the main characters and elaborate on their roles.
    d. Inter-Textual Links: Suggest tags for associating with other literary works and authors.

    Episodic Description According to Three-Act Structure:
    a. Act 1 - Setup: Provide a summary of the initial act, establishing the setting, characters, and the main conflicts.
    b. Act 2 - Confrontation: Describe the events and obstacles the main characters face, leading to the climax of the story.
    c. Act 3 - Resolution: Sum up how the story concludes, including how conflicts are resolved and the state of the characters.

    Content Assessment:
    a. Sentiment Analysis: Determine whether the sentiment in the text is positive, negative, or neutral, providing textual evidence. Rate the sentiment on a scale of -1 to 1.
    b. Destructive Content Detection: Check for any content promoting violence, discrimination, or hatred towards individuals/groups and provide supporting excerpts.

    Readability Metrics:
    a. Provide the Flesch Reading Ease score, Flesch-Kincaid Grade Level, and Gunning Fog Index.

    Political Orientation Analysis:
    a. Identify and explain liberal or conservative values, democratic or autocratic tendencies, and militaristic or humanistic themes in the text.
    b. Summarize the political orientation and rate on a scale of -1 to 1 for each dimension (Liberal-Conservative, Democratic-Autocratic, Militaristic-Humanistic).
    
    Result in Markdown format.
    Answer in 8000 words or less.
    """

    # Update session state
    instruct_text = st.text_area("Enter Instructions:", value=instruct_value, height=400)
    # st.session_state.messages.append()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for chat input if it doesn't already exist

prompt_disabled = (option == "Chat with PDF" and st.session_state['doc'] == "")

if prompt := st.chat_input("What is up?", disabled=prompt_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        processed_prompt = prompt
        if st.session_state['doc'] != "":
            processed_prompt = f"""I'm going to provide you with book in pdf file. 
            {instruct_text}
            
            Here are the book:
            <book>
             {st.session_state['doc']}
            </book>
            Answer the question immediately without preamble.
            {prompt}
            """

        result = model.predict(input=processed_prompt)

        for chunk in result:
            full_response += chunk
            time.sleep(0.01)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
