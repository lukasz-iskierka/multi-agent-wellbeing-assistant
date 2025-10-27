from src.utils.logging_utils import log
from src.schemas.models import Step, Steps
from src.schemas.states import AdvicePlanningState, PlanningOutputState

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
from termcolor import colored




def build_planner_subgraph():

    # Instatiate chat model
    llm_4o = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0) 

    # Nodes and edges

    advice_planner_instructions = """# Identity and objectives: 

    You are a wellbeing advisor tasked with planning a wellbeing action plan for a user. In doing so, you will be receiving a critical feedback from an another expert and the user. 

    Below you will find your conversation with the expert and the user.

    # Follow these instructions carefully:

    1. First, review the problem reported by the user:

    {problem}

    2. Review the current state of the conversation.
    3. Examine any optional feedback that has been provided by the expert in the conversation.      
    4. Examine any optional feedback that has been provided by the user in the conversation. 
    5. Preserve as unchanged those themes from the plan for which the expert and the user did not provide a critical feedback (didn't ask to delete or change).
    6. Plan up to {max_steps} steps (including the preserved ones) that could be taken to improve the user's wellbeing. 
    7. Each step should have its theme. Examples of themes (domains) for those steps include: 
    - home remedies,
    - conventional medicine,
    - lifestyle changes,
    - exercises,
    - therapy,
    - workplace adjustments, 
    - other themes (domains) that fit the problem reported by the user.
    8. In addition to the theme, each step should be accompanied with one helpful tip (one or two sentences). 
    8. You can provide up to three different tips for the same theme, but make sure to assign them to a separate instance of the theme.
    9. Here is an example of the expected wellbeing action plan:

    Proposed steps:

    - Step -
    Theme: Lifestyle changes 
    Helpful tip: Improving sleep quality can be effectively achieved by minimising exposure to blue light emitted by electronic devices, particularly in the hours leading up to bedtime.

    - Step -
    Theme: ...
    Helpful tip: ...

    10. Always try to improve the plan based on the feedback from both the user and the expert. If you think the plan cannot be improved any further, output the best version.
    11. Don't assume the role of the feedback provider. You are working on the feedback provided.
    """

    def advice_planner(state: AdvicePlanningState):
        
        """Advice-planning node"""
        
        cycles_counter = state.get("cycles_counter", -1)
        user_feedback = state.get("user_feedback", False)
        
        # Reset the cycles counter if user feedback was provided and re-work the plan
        if user_feedback: 
            cycles_counter = -1 
            user_feedback = False # reset to False

        # Print a progress message
        if cycles_counter == -1:
            log("[Planner] Drafting the plan...")

        problem = state["problem"]
        conversation = state.get("messages", [])
        max_steps = state.get("max_steps", 3)
        
        # Format the system message
        sys_message =  advice_planner_instructions.format(
            problem=problem,
            max_steps=max_steps
        )

        # Messages list
        messages = [
            SystemMessage(content=sys_message),
            AIMessage(content=f"Plan the wellbeing action plan for the user")
        ]
        
        # Trim the conversation if necessary
        if len(conversation) > 5:
            conversation = conversation[-5:]

        plan = llm_4o.invoke(messages + conversation)

        # Increment the counter
        cycles_counter += 1

        return {
            "messages": [plan], 
            "plan": plan.content,
            "cycles_counter": cycles_counter,
            "user_feedback": user_feedback
            }
    

    feedback_instructions = """# Identity and objectives: 
    You assuming a role of an expert at providing feedback for wellbeing action plans. You're known for your scrutiny and critical mindset; however, your feedback is always accurate and fair.

    # Follow these instructions carefully:
    1. First, review the problem reported by the user:

    {problem}

    2. Review your previous feedback (so you don't repeat yourself).
    3. Review the current version of the wellbeing action plan (the last message in the messages history).        
    4. Bear in mind that each step in the plan is intentionally kept short. All steps have their theme and are accompanied with a single helpful tip. 
    5. When evaluating the current version of the wellbeing action plan, consider these points:
    - Is the plan relevant to the the user's problem? Could anything be changed to increase the relevancy?
    - Considering the user's situation, are those helpful tips really helpful? Would adding alternatives help?
    - Are the themes in the plan diverse enough so that they approach the user's problem from different angles?
    - All helpful tips should be realistic and pragmatic. Perhaps the plan doesn't explore free alternatives?
    - Your feedback should be critical but constructive and actionable. 
    - Only focus on steps that could be improved and skip commenting on those that are ok.

    Example of your feedback:

    "Providing feedback for the following steps in the wellbeing action plan:

    - Step - 
    Theme: Exercises
    Helpful tip: Regular running can help reduce stress and improve overall mental wellbeing.

    My feedback: The explored theme is relevant as exercises can lower stress levels; however, running can not be apropriate for individuals with mobility issues. I suggest a gentler alternative, such as yoga or pilates.

    - Step - 
    Theme: ...
    Helpful tip: ..."

    4. Only output feedback for those steps that could use some additional work. 
    5. NEVER output revised version of the plan. Your feedback should limit to the specific steps and should be output in the "My feedback:" section (as seen in the example).
    6. Make sure not to repeat your previous feedback.  
    7. The plan shouldn't suggest it is a medical advice.
    8. CRUCIAL: If you think that no changes are required for the ENTIRE plan, and you approve it, output "No changes required for the plan." 

    """

    def feedback_generator(state: AdvicePlanningState):
        
        """Node providing feedback for the advice_planner."""

        problem = state['problem']
        conversation = state['messages']
        
        # Format the system message
        sys_message =  feedback_instructions.format(problem=problem)

        # Trim the conversation if necessary
        if len(conversation) > 5:
            conversation = conversation[-5:]

        feedback = llm_4o.invoke([SystemMessage(content=sys_message)] + conversation)
        feedback.name = "planner"

        return {'messages': [feedback]}


    def continue_planning(state: AdvicePlanningState):
        
        """Route based on the feedback from the feedback_generator node and completed |feedback| -> |planning| cycles"""

        cycles_counter = state['cycles_counter']
        max_cycles = state.get('max_cycles', 2) # defaults to 2 full |feedback| -> |planning| cycle
        conversation = state["messages"]

        # Route to the human feedback step if max cycles were reached
        if cycles_counter >= max_cycles:
        
            # Print progress log
            log("[Planner] Max cycles reached.")    
        
            return "human_feedback"
        
        # Route to the human feedback step if AI feedback generator approved the plan
        elif len(conversation) > 1 and conversation[-2].name == "planner" and "No changes required for the plan." in conversation[-2].content:
            
            # Print progress log
            log("[Planner] Draft successfully generated!")
            
            return "human_feedback"
        # Otherwise, continue with AI feedback generation 
        else:
            return "feedback_generator"
        
        
    def human_feedback(state: AdvicePlanningState):
        
        """Human feedback node"""

        plan = state["plan"]

        # Interrupt the graph execution, surface current version of the plan and return human input
        user_feedback = interrupt(f'\n\n* * * * *\n\nDo you have any suggestions for the proposed steps in your Wellbeing Action Plan?.\n\n{plan}\n\n* * * * *\n\n')

        # Print progress message
        log(f'[User input] Feedback from the user: "{user_feedback}"')
        
        if user_feedback =="No feedback":
            return {"user_feedback" : False}
        else:
            feedback_formatted = f"My (user) feedback:\n\n{user_feedback}"
            return {
                "user_feedback": True, 
                "messages": [HumanMessage(content=feedback_formatted, name="user")]
                }
        
        
    def act_on_feedback(state : AdvicePlanningState):
        
        """Route based on human feedback"""    

        user_feedback = state["user_feedback"]

        # If further work is required
        if user_feedback:
            return "advice_planner"
        else:
            return "plan_formatting"


    formatting_instructions = """# Identity and objectives: 

    You are an assistant tasked with formatting the wellbeing action plan into a set of steps. Each step should have its theme and helpful tip.

    # Follow these instructions carefully:

    1. Ignore the preamble and summary (if present)
    2. Ignore numbers (if present)
    3. Don't change the themes and helpful tips. 
    4. Your task is to output individual steps according to given schema while preserving the content of the steps.
    """

    def plan_formatting(state: AdvicePlanningState):
        
        """Node formatting the drafted plan steps based on expected output schema"""
        
        plan = state['plan']
        
        structured_llm = llm_4o.with_structured_output(Steps)
        structured_plan = structured_llm.invoke([formatting_instructions] + [AIMessage(content=plan)])

        return {"steps": structured_plan.steps}

    
    # Build the subgraph

    # Add nodes
    builder = StateGraph(state_schema=AdvicePlanningState, output_schema=PlanningOutputState)
    builder.add_node("advice_planner", advice_planner)
    builder.add_node("feedback_generator", feedback_generator)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("plan_formatting", plan_formatting)

    # Add edges (logic)
    builder.add_edge(START, "advice_planner")
    builder.add_conditional_edges("advice_planner", continue_planning, ["feedback_generator", "human_feedback"])
    builder.add_edge("feedback_generator", "advice_planner")
    builder.add_conditional_edges("human_feedback", act_on_feedback, ["plan_formatting", "advice_planner"])
    builder.add_edge("plan_formatting", END)

    # compile and return
    return builder.compile()


            