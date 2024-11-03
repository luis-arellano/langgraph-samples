from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field


class Reflection(BaseModel):
    """A reflection on ther research."""

    missing: str = Field(description="Critique what is missing")
    superfulous: str = Field(description="Critique what is superfulous")


class AnswerQuestion(BaseModel):
    """Answer the user's question."""

    answer: str = Field(description="~250 words detailed answer to the question")
    reflection: Reflection = Field(description="your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvement to address the critique of your current answer"
    )
