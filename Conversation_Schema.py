from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import LLMChain
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAPI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")


class Conversation_Schema:
    def __init__(self, vocab, session_id, user_id):
        self.prompt = None
        self.vocab = vocab
        self.memory = None
        self.history = None
        self.model = None
        self.session_id = session_id
        self.user_id = user_id
        self.langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            session_id=self.session_id,
            user_id=self.user_id
        )
    # Setters for each attribute

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_vocab(self, vocab):
        self.vocab = vocab

    def set_memory(self, memory):
        self.memory = memory

    def set_history(self, history):
        self.history = history

    def set_model(self, model):
        self.model = model

    # Getters for each attribute
    def get_prompt(self):
        return self.prompt

    def get_vocab(self):
        return self.vocab

    def get_memory(self):
        return self.memory

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model

    def init_conv(self):
        vocab = self.vocab

        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        self.history = ChatMessageHistory()

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a helpful AI that helps beginner English students practice their conversational skills.
                  You should only speak with them in simple sentences. Return your answer as a JSON that includes the following keys.
                  Analysis: Provides a detailed analysis of their answer and explains the grammatical reason why it is correct or incorrect as well as if it makes sense in the context of the conversation. Be helpful, cheerful, and encouraging.
                  Conversation: Respond to the student continuing the conversation based on what they said and any corrections you make in the Analysis. IMPORTANT! ONLY USE A SIMPLE BASIC SENTENCE IN YOUR RESPONSE. Focus on utilizing the following vocabulary words in your conversation: {vocab}
                  """),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template("{human_msg}")
        ])

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)

        return self.model, self.history, self.prompt, self.memory

    def call_llm(self, msg):
        chat_llm_chain = LLMChain(
            llm=self.model,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
        )

        response = chat_llm_chain.invoke(
            {"human_msg": msg}, config={"callbacks": [self.langfuse_handler]})

        return response


def translate_input(input):
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f""" Tranlate the following input into portugese with the exception of the words in paraenthesis.
                  """),
        HumanMessagePromptTemplate.from_template("{input_msg}")
    ])

    chain = prompt | model

    response = chain.invoke({"input_msg": input})

    return response


if __name__ == "__main__":

    conv = Conversation_Schema("sleep, eat, swim, beach, food, hungry")

    conv.init_conv()

    response = conv.call_llm("hello how are you")

    print(response)
