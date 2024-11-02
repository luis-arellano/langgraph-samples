from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from typing import Sequence
import logging

from langchain_openai import ChatOpenAI

#Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Creating the prompts. 1st for the reflection, 2nd for the generation
reflection_prompt = ChatPromptTemplate.from_messages(
   [(
       "system",
       "You are a viral twitter influencer grading a tweet. Generate critique and recomeendations for the user"
       "Always provide detailed recommendations, including lenght, virality, style."
   ),
    MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
    (
       "system",
       "You are a twitter techie influencer assistant tasked with generating excellent tweeter posts. "
       "Generate the best tweet based on the user's request. "
       "If the user provides critique, provide a revised version of your previous attempt. "
       "Do not include any explanations - just the tweet text."
       
   ),
    MessagesPlaceholder(variable_name="messages")
    ]
)


# now create the chains
llm = ChatOpenAI(model="gpt-4-turbo")
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm



# Define constants, to be used as node names in the graph
REFLECT = "reflect"
GENERATE = "generate"

# Crete the langgraph nodes

def generation_node(state: Sequence[BaseMessage]):
    logger.info("Generating tweet")
    return generation_chain.invoke({"messages": state})

def reflection_node(state: Sequence[BaseMessage]):
    logger.info("Reflecting on tweet")
    res = reflection_chain.invoke({"messages": state})
    
    # We take the content of the response and make it a human message
    # in order to trick the LLM into thinking that the user has replied
    # which will create a conversation loop
    return [HumanMessage(content=res.content)]

# Now we create the graph. Where we define the nodes and edges and conditional paths
builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# this is a conditioanl step that is hardcoded, but we could use another LLM on this 
# conditional step if we wanted to.
def should_continue(state: Sequence[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# To visualize the graph
# print(graph.get_graph().draw_mermaid()) # take the output and paste it into mermaid.live to visualize
graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello, LangGraph!")
    
    # RUN THE GRAPH
    input = HumanMessage(content="""
                         Make this tweet better:" "
                         
                        ChatBots are overated use of RAG. You can create a completely automated RAG powered workflow.
                        If you have an army of BPOs doing repetive processes. You can set up a workflow to repetealy call a RAG flow
                        and then have this continuously running operations.
                        This idea can safe you or generate you millions of dollars.
                         """)
    response = graph.invoke(input)
    
    
    for msg in response:
        print('RESPONSE:', msg.content)
