# reference (Whisper)
## https://towardsdatascience.com/transcribe-audio-files-with-openais-whisper-e973ae348aa7

# reference (OpenAI)
## https://platform.openai.com/docs/guides/speech-to-text/quickstart
## https://digiconfactory.tistory.com/entry/파이썬-OpenAI-API-whisper-사용하기

# reference (HF wav2vec2)
## https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean
## https://colab.research.google.com/github/indra622/tutorials/blob/master/wav2vec2_korean_tutorial.ipynb#scrollTo=ykdQ2umfcqZQ

# reference 
## https://velog.io/@acdongpgm/ChatGPT-LangChain을-이용한-Linkendin-이력서-요약-질문하기

import os
import argparse
import openai
import librosa
import streamlit as st
import soundfile as sf

import ffmpeg
from PIL import Image
from pytube import YouTube
from copy import deepcopy
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA



def arg_parser():
    # Arguement parsing..
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, default='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', help="If you don't know key value, Just ask jskim")
    args = parser.parse_args()
    return args


def text_custom(font_size, text):
    '''
    font_size := ['b', 'm', 's']
    '''
    result=f'<p class="{font_size}-font">{text}</p>'
    return result




def main(args):

    st.set_page_config(layout='wide', page_title=f'STT & AI Analysis Demo by AIC :sunglass:')

    # reference
    ## https://discuss.streamlit.io/t/change-input-text-font-size/29959/4
    ## https://discuss.streamlit.io/t/change-font-size-in-st-write/7606/2
    st.markdown("""<style>.b-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)    
    st.markdown("""<style>.m-font {font-size:20px !important;}</style>""" , unsafe_allow_html=True)    
    st.markdown("""<style>.s-font {font-size:15px !important;}</style>""" , unsafe_allow_html=True)    
    tabs_font_css = """<style>div[class*="stTextInput"] label {font-size: 15px;color: black;}</style>"""
    st.write(tabs_font_css, unsafe_allow_html=True)

    st.title('STT & ChatGPT Demo by AIC')
    st.header('Phase 1: STT process')

    with st.sidebar:
        chunk_size = st.slider(
            "Chunk size",
            0, 1500, 500,
        )

        overlap_size = st.slider(
            "Overlap size",
            0, 500, 150,
        )

        stt_model = st.selectbox(
            label='STT Model',
            options=['whisper-1']
        )

        embeddeing_model = st.selectbox(
            label='Embedding Model',
            options=['text-embedding-ada-002']
        )

        summarization_model = st.selectbox(
            label='Summarization LLM Model',
            options=['gpt-3.5-turbo',
                     'text-davinci-003']

        )

        qa_model = st.selectbox(
            label='QA LLM Model',
            options=['gpt-3.5-turbo',
                     'text-davinci-003']

        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.01,
        )

        chain = st.radio(
            label='Chain type',
            options=['stuff',
                    'map_reduce']
        )

        st.markdown(
            """
            **Blog post:** \n
            [*Blog title*](URL)

            **Code:** \n
            [*Github*](https://github.com/jskim0406/SimpleRec_w_langchain.git)
            """
        )

    t = '사용할 모델은 OpenAI의 Whisper라는 모델입니다. STT 분야에서는 가장 좋은 성능을 보이는 대표적인 모델입니다.'
    st.markdown(text_custom('m', t), unsafe_allow_html=True)

    image = Image.open('img/whisper_overview.png')
    st.image(image, caption="Whisper Architecture(From 'OpenAI Whisper Github')")

    t = '아래 영상의 오디오(speech)에서 Text를 추출할 것입니다.'
    st.markdown(text_custom('m', t), unsafe_allow_html=True)
    youtube_url = st.text_input(label="Youtube URL을 입력해주세요.",
                                value='https://www.youtube.com/watch?v=9lRv6i3efNg&t=52s')

    if youtube_url:
        width = 40
        side = max((100 - width) / 2, 0.01)
        _, container, _ = st.columns([side, width, side])
        container.video(data=youtube_url)
    else:
        t = 'Youtube URL을 입력하면 영상이 Embedding됩니다. :)'
        st.markdown(text_custom('s', t), unsafe_allow_html=True)

    st.subheader('Phase 1-1: Audio Extraction from Youtube video')
    t = '그 전에 우선 Youtube Video에서 Audio를 추출할 것입니다.'
    st.markdown(text_custom('m', t), unsafe_allow_html=True)
    t = "Youtube에서 Audio를 추출하기 위해 'Pytube'라는 python 패키지를 활용합니다."
    st.markdown(text_custom('s', t), unsafe_allow_html=True)

    if st.button("Get Audio. START !"):
        with st.spinner('Audio Extraction in progress..'):
            selected_video = YouTube(youtube_url)

            stream_url = selected_video.streams.all()[0].url  # Get the URL of the video stream

            audio, err = (
                ffmpeg
                .input(stream_url)
                .output("pipe:", format='wav', acodec='pcm_s16le')  # Select WAV output format, and pcm_s16le auidio codec. My add ar=sample_rate
                .run(capture_stdout=True)
                )

            audio_f_name = 'audio_test'
            with open(audio_f_name+'.mp3', 'wb') as f:
                f.write(audio)
        st.success(f'Audio extraction & and Saved in "{audio_f_name}" is Done!')

        if audio:
            st.audio(audio_f_name+'.mp3')

    
    global txt
    txt = None

    st.subheader('Phase 1-2: Speech to Text')
    if st.button("STT. START !"):
        # STT with OpenAI API
        openai.api_key = args.apikey
        audio_f_name = 'audio_test'
        audio_file = open(f'{audio_f_name}.mp3', 'rb')

        with st.spinner('STT in progress..'):
            transcript = openai.Audio.transcribe(stt_model, audio_file)
        txt = transcript['text']

        text_f_name = 'extracted.txt'
        with open("extracted.txt", "w") as f:
            f.write(txt)
        st.success(f'STT is Done & Saved in "{text_f_name}"!')

        with st.expander("Extracted texts..", expanded=True):
            txt


    st.header('Phase 2: AI Analysis process')
    st.markdown('사용할 모델은 OpenAI의 ChatGPT입니다. 아시는 것과 같이 최근 성능과 활용성 면에서 가장 주목받는 AI 모델입니다.')

    st.subheader('Phase 2-1: Text summarization')
    text_f_name = 'extracted'
    if os.path.isfile(f"{text_f_name}.txt"):
        with st.expander("Extracted texts..", expanded=False):
            f = open(f"{text_f_name}.txt", "r")
            txt_ext= f.read()
            txt_ext
            f.close()

    if st.button("Summarization. START !"):
        with st.spinner('OpenAI model loading..'):
            if summarization_model == 'gpt-3.5-turbo':
                # Chat
                # Reference: https://towardsdatascience.com/summarizing-the-latest-spotify-releases-with-chatgpt-553245a6df88
                llm = ChatOpenAI(temperature=temperature, model_name=summarization_model, openai_api_key=args.apikey)
                st.success(f'OpenAI model("{summarization_model}") is loaded.')
            else:
                # LLM
                # Reference: https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html#:~:text=refine_prompt%3Drefine_prompt
                llm = OpenAI(temperature=temperature, model_name=summarization_model, openai_api_key=args.apikey)
                st.success(f'OpenAI model("{summarization_model}") is loaded.')

        with st.spinner('Chunking is in progress..'):
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                separator='.',
                chunk_overlap=overlap_size
            )
            texts = text_splitter.split_text(txt_ext)
            docs = [Document(page_content=t) for t in texts]
        st.success(f'Chunking is Done.')

        with st.spinner('Summarization is in progress..'):
            if summarization_model == 'gpt-3.5-turbo':     
                prompt_template = """Write a comprehensive summary about the given texts in KOREAN:\n\n{text}."""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
                if chain == 'stuff':
                    chain = load_summarize_chain(llm, chain_type=chain, prompt=PROMPT, verbose=True)
                elif chain == 'map_reduce':
                    chain = load_summarize_chain(llm, chain_type=chain, map_prompt=PROMPT, combine_prompt=PROMPT, verbose=True)
            else:
                prompt_template = """Write a comprehensive summary about the given texts in KOREAN:
                {text}
                """
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
                if chain == 'stuff':
                    chain = load_summarize_chain(llm, chain_type=chain, prompt=PROMPT, verbose=True)                    
                elif chain == 'map_reduce':
                    chain = load_summarize_chain(llm, chain_type=chain, map_prompt=PROMPT, combine_prompt=PROMPT, verbose=True)
            sum_txt = chain.run(docs)
        st.success(f'Summarization is Done.')

        with st.expander("Show me summarized text..", expanded=True):
            sum_txt

        with open("summarized_extracted.txt", "w") as f:
            f.write(sum_txt)


    st.subheader('Phase 2-2: QA based on extracted text')
    if os.path.isfile("extracted.txt"):
        with st.expander("Extracted texts..", expanded=False):
            f = open("extracted.txt", "r")
            txt_ext= f.read()
            txt_ext
            f.close()

    if os.path.isfile("summarized_extracted.txt"):
        with st.expander("Summarized texts..", expanded=False):
            f = open("summarized_extracted.txt", "r")
            txt_ext= f.read()
            txt_ext
            f.close()

    global query
    query = st.text_input('질문을 입력해주세요.',
                            placeholder=f'Ask me anything from your text.'
                            )
    
    if query:
        if st.button("Question Answering. START !"):
            with st.spinner('Chunking is in progress..'):
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    separator='.',
                    chunk_overlap=overlap_size
                )
                texts = text_splitter.split_text(txt_ext)
                docs = [Document(page_content=t) for t in texts]
            st.success(f'Chunking is Done.')

            with st.spinner('OpenAI Embedding is loading & Embedding in progress..'):
                embeddings = OpenAIEmbeddings(openai_api_key=args.apikey, 
                                              model=embeddeing_model)
                docsearch = Chroma.from_documents(docs, embeddings)
            st.success(f'Embedding is Done. VectorDB is created.')

            with st.spinner('Question Answering functionality is in progress..'):
                if qa_model == 'gpt-3.5-turbo':
                    model_name=qa_model
                    llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=args.apikey)
                else:
                    model_name=qa_model
                    llm = OpenAI(temperature=temperature, model_name=model_name, openai_api_key=args.apikey)
                qa = RetrievalQA.from_chain_type(llm=llm, 
                                                chain_type=chain, 
                                                retriever=docsearch.as_retriever())
                st.success(f'Question Answering is ready.')
        
                with st.spinner('Please wait for the answer :) '):
                    query += ' 한국어로 답변해주세요. Answer in KOREAN.'
                    ansewr = qa.run(query)
                st.success(f'Done.')

                with st.expander("Answer is ..", expanded=False):
                    ansewr


if __name__=='__main__':
    args = arg_parser()
    main(args)