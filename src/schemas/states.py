from src.schemas.models import Step
from langgraph.graph import MessagesState
from typing_extensions import TypedDict
from typing import Annotated, List
import operator

# Advice Planning subgraph state
class AdvicePlanningState(MessagesState):
    problem: str # problem reported by a user
    max_steps: int # maximum number of steps in the wellbing action plan
    max_cycles : int # maximum number of allowed |ai_feedback| -> |advice_planner| cycles
    plan: str # a wellbeing action plan that has been approved by the user
    steps: List[Step] # a list of selected steps in the wellbeing action plan
    cycles_counter : int # for tracking the cycles
    user_feedback : bool # if user provided a feedback to work on

# Advice Planning subgraph output state
class PlanningOutputState(TypedDict):
    problem: str # problem reported by a user
    steps: List[Step] # a list of selected steps in the wellbeing action plan
    max_cycles : int # maximum number of allowed |ai_feedback| -> |advice_planner| cycles

# Consultation subgraph state
class ConsultationState(MessagesState):
    problem: str # user-reported issue
    max_cycles: int # Max number of |question| -> |answer| cycles
    step: Step # an individual step from the plan received through Send() API
    webquery: str # a query constructed for the web search
    wikiquery: str # a query constructed for the Wikipedia search
    transcript: str # transcript from the consultation
    summary: str # summary of the consultation (for exceptionally long lists of messages)
    cycles_counter : int # |question| -> |answer| cycles counter 
    source_docs: Annotated[list, operator.add] # docs with the context the practitioner is using to provide answers
    sections: list # Written section aggregated in the OverallState through Send() API

class ConsultationOutputState(TypedDict):
    sections: list # Written section aggregated in the OverallState through Send() API  

# Parent graph state
class OverallState(TypedDict):
    problem: str # problem reported by a user
    max_steps: int # maximum number of steps in the wellbing action plan
    max_cycles: int # depth of research across both subgraphs (advice planner and consultation) 
    sections: Annotated[list, operator.add] # Send() API key where all written sections are aggregated
    final_plan: str # Final version of the plan including all individual sections