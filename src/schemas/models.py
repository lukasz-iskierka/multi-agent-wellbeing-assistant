from typing import Annotated, List
from pydantic import BaseModel, Field

# Schema for the Advice Planning subgraph
class Step(BaseModel):
    theme: str = Field(
        description="A theme which could be explored to improve the person's wellbeing in the context of the problem that they presented."
        )
    helpful_tip: str = Field(
        description="A piece of advice which follows the theme."
    )

    @property
    def step_summary(self) -> str:
        return f"Theme: {self.theme}\n\nHelpful tip: {self.helpful_tip}"

class Steps(BaseModel):#
    steps : List[Step] = Field(
        description="A list of steps which could be taken to improve user's wellbeing"
    )

    # Schema for the search query formatting
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")