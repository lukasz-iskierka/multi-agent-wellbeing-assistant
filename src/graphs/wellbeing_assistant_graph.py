from src.graphs.subgraphs.advice_planning_subgraph import build_planner_subgraph 
from src.graphs.subgraphs.consultation_subgraph import build_consultation_subgraph
from src.schemas.models import Step
from src.schemas.states import OverallState, PlanningOutputState
from src.utils.logging_utils import log, init_timer
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, AIMessage


def build_main_graph():

    # Instantiate chat model
    llm_5_mini = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)

    # Dynamic parallelisation logic (mapping step of the Map-Reduce workflow)
    def map_to_consultation(state: PlanningOutputState):
        
        """Conditional edge to map each step in the plan to an individual instance of the consultation_subgraph."""
        
        # Print progress message
        log("[Consultation] Searching evidence for the drafted plan...")

        steps = state['steps']
        problem = state["problem"]
        max_cycles = state.get("max_cycles", 2)

        # Map each step
        return [Send("consultation_subgraph", {"step": step,
                                            "problem": problem,
                                            "max_cycles": max_cycles}) for step in steps]


    plan_writer_instructions = """# Identity an objectives:
    You are an expert technical writer creating a polished version of a Wellbeing Action Plan. 
    You are presented with pre-written individual sections of the plan, each focusing on a different actionable step a client can take to deal with the problem they have reported:

    ## Their problem:

    {problem}

    ## Pre-written sections:

    {all_sections}

    ---

    # Follow these steps:    
    1. Review the pre-written sections for each step of the plan.
    2. Consolidate these sections into one Wellbeing Action Plan. 
    3. Aim to keep the individual sections unchanged, unless you think their style/flow could be improved wthout loosing their meaning and detail.
    4. Check if the sections don't duplicate one another. If the do, shift focus to another aspect of the pre-written section so it is more distinct form other sections.
    4. Include the title headers for each section while excluding the `### Summary` headers.
    5. There should be only one Summary section in the finished plan with a brief outline of all the steps.
    6. Use markdown formatting. 
    7. Start the plan with a single title header: `# Personalised Wellbeing Action Plan`
    8. Preserve any citations in the sections, which will be annotated in brackets, for example [1] or [2]
    9. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
    10. List your sources in order and do not repeat.
    11. Include no pre-amble for the plan. Only output the finished plan.
    12. If you format some steps as a numbered list, be consistent and do the same for other sections (if appropriate).

    12. Expected structure of the plan:

    # Personalised Wellbeing Action Plan

    ## Summary

    Body of text for the summary.

    ## Section header (with actual title)

    Body of text for the section.

    ...

    ### Sources

    [1] Source 1
    [2] Source 2 
    ...

    """ 

    # Final node
    def plan_writer(state: OverallState):
        
        """Node writing the final version of the wellbeing action plan."""
        
        # Print progress message
        log("[Finalising] Writing final version of personalised wellbeing action plan...")

        problem = state["problem"]
        sections = state["sections"]

        # Format the instructions
        formatted_plan_instructions = plan_writer_instructions.format(
            problem=problem,
            all_sections="\n\n---\n\n".join([section for section in sections]) # Format all pre-written sections
        )

        messages = [
            SystemMessage(content=formatted_plan_instructions),
            AIMessage(content="Write a finished version of the Wellbeing Action Plan")
        ]

        # Generate the final plan
        final_plan = llm_5_mini.invoke(messages)

        # Print progress message
        if final_plan.content:
            log("[Completed] Plan successfully generated!")

        return {"final_plan": final_plan.content}


    # Build the parent graph
    builder = StateGraph(OverallState)
    
    # Create subgraphs
    planner_subgraph = build_planner_subgraph()
    consultation_subgraph = build_consultation_subgraph()
    
    # Add nodes (subgraphs)
    builder.add_node("advice_planning_subgraph", planner_subgraph)
    builder.add_node("consultation_subgraph", consultation_subgraph)
    builder.add_node("plan_writer", plan_writer)

    # Add logic
    builder.add_edge(START, "advice_planning_subgraph")
    builder.add_conditional_edges("advice_planning_subgraph", map_to_consultation, ["consultation_subgraph"])
    builder.add_edge("consultation_subgraph", "plan_writer")
    builder.add_edge("plan_writer", END)

    # Include memory
    memory = MemorySaver()
    
    # Compile and return the main graph
    return builder.compile(checkpointer=memory)


