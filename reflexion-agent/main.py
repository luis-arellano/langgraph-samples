from dotenv import load_dotenv
import logging
import datetime
import json
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, Reflection

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher..
     Current time: {time}
     1. {first_instruction}
     2. Reflect and critique your answer. Be severe to maximize improvement.
     3. You must recommend search queries to research information and improve your answer.
     
     Your response MUST include both an answer and search queries.
     """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())


# LLM
llm = ChatOpenAI(model="gpt-4-turbo")

# Output parser
parser = JsonOutputToolsParser(
    return_id=True
)  # returns the funciton call and transform it into a dict
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words answer."
)

# This forces the LLM to always use the AnswerQuestion tool, thus grounding the response to the schema we want
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

if __name__ == "__main__":
    print("Hello, Reflexion Agent!")
    
    # Intial prompt
    human_message = HumanMessage(
        content="Write about AI-powered SOC / autonomous SOC problem domain"
        "list startups that do that and raise capital"
        )
    
    # # similar to the first responder chain, but with the pydantic parser
    # # so that it takes the response and parses it into the AnswerQuestion pydantic object
    chain = (
        first_responder_prompt_template | llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    
    res = chain.invoke(input={"messages": [human_message]})
    
    # Print using json jump to make it more readable
    print(json.dumps(res[0].dict(), indent=2, ensure_ascii=False))
    
    
     #### TO Understand the prompt and LLM configuration #############

    # formatted_prompt = first_responder_prompt_template.invoke({"messages": [human_message]})
    # print("\nFormatted Prompt:")
    # print(formatted_prompt)

    # # See what the LLM will receive including function definitions
    # llm_with_tools = llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    # print("\nLLM Configuration:")
    # print(llm_with_tools)

    # # You can then continue the execution
    # llm_response = llm_with_tools.invoke(formatted_prompt)
    # print('\nllm_response: ', llm_response)
    # final_response = parser_pydantic.invoke(llm_response)
    # print('\nfinal_response: ', final_response)