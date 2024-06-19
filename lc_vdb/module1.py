# %% Imports

from dotenv import load_dotenv
from langchain.llms import OpenAI

# %% Example 1 - Call an LLM ############################################


llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
# %%

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoors activities."
print(llm(text))


# %% Example 2 - A CHAIN ############################################

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# %%
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# %% Run the chain
print(chain.run(product="eco-friendly water bottles"))

# %% Example 3 - Memory ############################################

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(),
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# display the conversation
print(conversation)


# %% Example 4 - DeepLake ############################################
import os

from dotenv import load_dotenv

load_dotenv()
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

# %% Initilaize the LLM and embeddings models
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# %% create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769.",
    "Louis XIV was born in 5 September 1638.",
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# %% create Deep Lake dataset
activeloop_org_id = os.getenv("ACTIVLOOP_ORG_ID")
activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# %%
db.add_documents(docs)
# %% Retrieval QA cain as a tool:
from langchain.agents import AgentType, Tool, initialize_agent

retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions.",
    )
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# %% Use the agent to ask question.

response = agent.run("When was Napoleon born?")
print(response)

# %%
