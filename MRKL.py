from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchainDemo.constants import openai_key, serp_api_key
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"]=serp_api_key

#The search tool has no async implementation, we fallback to sync
@cl.langchain_factory(use_async=False)
def load():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="This is useful when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="This is useful when you need to answer questions about math"
        ),
    ]
    return initialize_agent(
        tools, llm1, agent="chat-zero-shot-react-description", verbose=True
    )