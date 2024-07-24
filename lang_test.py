
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import LLMChain
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

history = ChatMessageHistory()

vocab = "swim,sleep,eat"

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=f"""You are a helpful AI that helps beginner english students practice their conversational skills.
                  You should only speak with them in simple sentences. Return your answer as a JSON that includs the following keys.
                  Analysis: Provides and analysis about their answer and explain the grammatical reason why it is correct or incorrect. Be helpful, cheerful, and encouraging.
                  Converstaion: Respond to the student continuing the converstation based on what they said and any corretions you make it the Analysis. IMPORTANT! ONLY USE A SIMPLE BASIC SENTENCE IN YOUR RESPONSE. Focus on the utalizing the following vocabulary words in your conversation:{vocab}
                  """),
    MessagesPlaceholder(
        variable_name="chat_history"
    ),
    HumanMessagePromptTemplate.from_template("{human_msg}")
]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

chat_openai = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="sk-proj-1Ma9ih5WYF8xBfvy9bZ4T3BlbkFJLmpOxoDLKmNuXAdeawRw",
    model_kwargs={"response_format": {"type": "json_object"}},

)


chat_llm_chain = LLMChain(
    llm=chat_openai,
    prompt=prompt,
    verbose=True,
    memory=memory,
)


print(prompt)
response = chat_llm_chain.invoke(
    {"human_msg": "Hi there my friend"})

print(response["text"])

response2 = chat_llm_chain.invoke(
    {"human_msg": "Yes I like the pool"})

print(response2["text"])

print(memory)
