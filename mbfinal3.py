import os
import pandas as pd
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import pygame
import re
import time
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

WAKE_WORDS = [
    'hey nova',
    'hi nova',
    'hello nova',
    'hoi nova',
    'nova',
    'innova',
    'inova',
    'hey innova',
    'hi innova',
    'hello innova',
    'hey inova',
    'hi inova',
    'hello inova',
    'no va',
    'noah',
    'hey noah',
    'hi noah',
    'hello noah',
    'noba',
    'hey noba',
    'hi noba',
    'hello noba',
    'nava',
    'hey nava',
    'hi nava',
    'hello nava',
    'novaa',
    'hey novaa',
    'hi novaa',
    'hello novaa'
]

# Load API Key
load_dotenv('.env.local')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Voice input setup
recognizer = sr.Recognizer()

def get_voice_input(filename='output.wav', duration=4, samplerate=44100, log_func=None):
    if log_func:
        log_func("\n🎙️ Listening... Speak now.")
    else:
        print("\n🎙️ Listening... Speak now.")

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            if log_func:
                log_func(f"🗣️ You said: {text}")
            else:
                print("You said:", text)
            return text
        except sr.UnknownValueError:
            if log_func:
                log_func("❌ Could not understand audio.")
            else:
                print("❌ Could not understand audio.")
        except sr.RequestError as e:
            if log_func:
                log_func(f"⚠️ Recognition service error: {e}")
            else:
                print("⚠️ Recognition service error:", e)
    return None

def clean_text_for_speech(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"[#*_~><|\\/\[\]{}]", "", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

DOMAIN_KEYWORDS = [

    # Original keywords and their spelling variations
    'maker bhavan', 'maker bhawan','maker bhaven','inventics','invent x','maker bhavans',
    'iitgn', 'iit gandhinagar', 'iit gandhi nagar', 'iitg',
    '3d printing', '3d print', '3d printer',
    'workshop', 'workshops',
    'faculty lead', 'faculty leader',
    'shivang sharma',
    'abhi raval', 'abhi rawal','abhiii raval','abhii raval',
    'pratik mutha','prateek mutha','prateek muttha','pratik muttha','pratik','prof mutha','prateek mutta','pratik mutha','professor mutha','professor pratik mutha',
    'aniruddh mali', 'anirudh mali',
    'invention factory','divij yadav','divij yadavs','divij',
    'innovation', 'innovations',
    'prototyping', 'prototype', 'prototypings',
    'inventx','madhu vadali'
    'tinkerers lab',
    'vishwakarma award',
    'leap program',
    'skill builder', 'skills builder',
    'sprint workshop', 'sprint workshops',
    'summer student fellowship',
    'industry engagement',
    'maker competition', 'maker competitions',
    'electronics prototyping', 'electronics prototyping zone', 'electronic prototyping',
    'pcb milling', 'pcb mill',
    'metal 3d printing',
    'fused deposition modeling',
    'sla printing', 'sla print',
    'laser cutting',
    'vacuum forming',
    'cnc', 'cnc machine',
    'digital fabrication', 'digital fabricate',
    'interactive design lounge',
    'collaborative classroom',
    'project-based learning',
    'active learning',
    'experiential education',
    'reverse engineering',
    'safety training',
    'project officer', 'project offcer',
    'project management', 'project manager',
    'mentorship',
    'incubation', 'incubator',
    'startup support', 'startup supports',
    'student startups',
    'patent filing', 'patent file',
    'innovation challenge',
    'maker culture', 'maker cultures',
    'hands-on learning', 'hands on learning',
    'hemant kanakia',
    'damayanti bhattacharya',
    'maker bhavan foundation',
    'science awareness program', 'science awareness',
    'indian academic makerspaces summit', 'iam summit',
    'center for essential skills',
    'design thinking',
    'entrepreneurship',
    'collaboration', 'collaborations',
    'interdisciplinary learning',
    'mechanical fabrication',
    'equipment booking',
    'technology transfer office', 'tech transfer office',
    'project proposal',
    'industry collaboration',
    'faculty collaboration', 'faculty collaborate',
    'mentor',
    'patron',

    # Related keywords and their spelling variations
    'rapid prototyping', 'rapid prototype', 'rapid-prototyping',
    'additive manufacturing', 'additive-manufacturing',
    'makerspace', 'maker space', 'makerspaces', 'maker spaces',
    'innovation lab', 'innovation laboratory', 'innovation-lab',
    'design sprint', 'design sprints', 'design-sprint',
    'creative workshop', 'creative workshops',
    'product design', 'product designing',
    'engineering design', 'engineering designing',
    'robotics', 'robotic',
    'automation', 'automate',
    '3d scanning', '3d scan', '3d-scanning',
    'laser engraving', 'laser engrave', 'laser-engraving',
    'circuit design', 'circuit designing',
    'microcontroller programming', 'microcontroller program',
    'arduino', 'arduinos',
    'raspberry pi', 'raspberrypi', 'raspi', 'raspberry pies',
    'iot', 'internet of things',
    'embedded systems', 'embedded system',
    'startup incubation', 'startup incubator', 'startup-incubation',
    'venture funding', 'venture fund', 'venture-funding',
    'technology commercialization', 'tech commercialization',
    'patent application', 'patent applications',
    'research collaboration', 'research collaborations',
    'skill development', 'skills development',
    'technical training', 'technical trainings',
    'maker community', 'makers community', 'makers’ community',
    'open source hardware', 'open-source hardware',
    'hardware hacking', 'hardware hack',
    'design innovation', 'design innovations',
    'prototype testing', 'prototype test',
    'user experience design', 'ux design', 'user-experience design',
    'human-centered design', 'human centred design', 'human centered design',
    'sustainable design', 'sustainability design'

    'maker bhavan', 'iitgn', '3d printing', 'workshop', 'faculty lead','maker bhawan',
    'shivang sharma', 'abhi raval', 'abhi rawal', 'pratik mutha', 'aniruddh mali', 'anirudh mali',
    'invention factory', 'prototyping', 'inventx', 'tinkerers lab',
    'vishwakarma award', 'leap program', 'skill builder', 'sprint workshop',
    'summer student fellowship', 'industry engagement', 'maker competition',
    'electronics prototyping', 'pcb milling', 'metal 3d printing',
    'fused deposition modeling', 'sla printing', 'laser cutting',
    'vacuum forming', 'cnc', 'digital fabrication', 'interactive design lounge',
    'collaborative classroom', 'project-based learning', 'active learning',
    'experiential education', 'reverse engineering', 'safety training',
    'project officer', 'project management', 'mentorship', 'incubation',
    'startup support', 'student startups', 'patent filing', 'innovation challenge',
    'maker culture', 'hands-on learning', 'hemant kanakia',
    'damayanti bhattacharya', 'maker bhavan foundation', 'science awareness program',
    'indian academic makerspaces summit', 'iam summit', 'center for essential skills',
    'design thinking', 'entrepreneurship', 'collaboration',
    'interdisciplinary learning', 'mechanical fabrication',
    'electronics prototyping zone', 'equipment booking',
    'technology transfer office', 'project proposal', 'industry collaboration',
    'faculty collaboration', 'mentor', 'patron'
]

def process_jsonl_to_documents(file_path):
    documents = []
    data = pd.read_json(file_path, lines=True)
    for _, row in data.iterrows():
        system_prompt = row['messages'][0]['content']
        question = row['messages'][1]['content']
        answer = row['messages'][2]['content']
        documents.append(Document(
            page_content=f"Question: {question}\nAnswer: {answer}",
            metadata={"source": "maker_bhavan_dataset", "topic": system_prompt}
        ))
    return documents

class ImprovedChatbot:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback 
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question"
        )
        self.vectorstore = self.initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        self.rag_chain = self.create_rag_chain()
        self.general_chain = self.create_general_chain()

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def initialize_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db_path = "./chroma_db"
        if os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            docs = process_jsonl_to_documents("dataset.jsonl")
            return Chroma.from_documents(docs, embeddings, persist_directory=db_path)

    def create_rag_chain(self):
        prompt = PromptTemplate.from_template(
            "You are Maker Bhavan's official assistant. Use this context below, respond ** clearly and concisely ** :\n"
            "Context: {context}\n"
            "Chat History: {chat_history}\n"
            "Question: {question}\n"
            "Answer:"
        )
        return (
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.get_relevant_documents(x["question"]),
                chat_history=lambda x: self._format_chat_history(x.get("chat_history", []))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def create_general_chain(self):
        prompt = PromptTemplate.from_template(
            "You are a helpful assistant. Give a brief and relevant answer to each question.\n"
            "{chat_history}\n"
            "User: {question}\n"
            "Assistant:"
        )
        return (
            RunnablePassthrough.assign(
                chat_history=lambda x: self._format_chat_history(x.get("chat_history", []))
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_chat_history(self, chat_history):
        return "\n".join(
            f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}"
            for msg in chat_history[-8:]
        )

    def is_domain_query(self, query):
        return any(keyword in query.lower() for keyword in DOMAIN_KEYWORDS)

    def speak(self, text):
        cleaned_text = clean_text_for_speech(text)
        self._log("🔊 Speaking...")
        try:
            tts = gTTS(cleaned_text, slow=False, lang='hi', tld='co.in')
            tts.save("response.mp3")

            pygame.mixer.init()
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pass

            pygame.mixer.music.unload()
            os.remove("response.mp3")
        except Exception as e:
            self._log(f"❌ gTTS playback error: {str(e)}")

    def chat_interface(self, require_wake_word=True):
        self._log("🤖 Enhanced Hybrid Chatbot Initialized (Say 'exit' to quit)")
        if require_wake_word:
            while True:
                self._log("\n🔍 Waiting for wake word...")
                query = get_voice_input(log_func=self._log)
                if not query:
                    continue
                if query.lower() == 'exit':
                    self._log("👋 Goodbye!")
                    break
                if any(wake_word in query.lower() for wake_word in WAKE_WORDS):
                    self.speak("Hello! How can I assist you?")
                    self.active_conversation()
                    break
                else:
                    self._log("👂 Waiting for the correct wake word...")
        else:
            self.active_conversation()

    def active_conversation(self):
        session_active = True
        session_start_time = time.time()
        while session_active:
            self._log("\n🎙️ Listening for your query...")
            query = get_voice_input(log_func=self._log)
            if not query:
                if time.time() - session_start_time > 30:
                    self.speak("Do you want to continue the conversation?")
                    response = get_voice_input(log_func=self._log)
                    if response and any(word in response.lower() for word in ['no', 'exit', 'bye']):
                        self.speak("Have a good day!")
                        session_active = False
                        break
                    else:
                        session_start_time = time.time()
                continue

            if query.lower() in ['exit', 'bye', 'goodbye','no']:
                self.speak("Have a good day!")
                session_active = False
                break

            try:
                memory_data = {
                    "question": query,
                    "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
                }
                if self.is_domain_query(query):
                    self._log("\nBot (RAG):")
                    response = self.rag_chain.invoke(memory_data)
                else:
                    self._log("\nBot (General):")
                    response = self.general_chain.invoke(memory_data)

                self.memory.save_context({"question": query}, {"answer": response})
                self._log(response)
                self.speak(response)
                session_start_time = time.time()
            except Exception as e:
                self._log(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    chatbot = ImprovedChatbot()
    chatbot.chat_interface()
