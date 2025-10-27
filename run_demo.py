# Load environment variables
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Imports
from src.graphs.wellbeing_assistant_graph import build_main_graph
from src.utils.logging_utils import init_timer
import os
from termcolor import colored 
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme


def main(initial_input):
    
    # Verify if all environment variables are loaded
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"] 
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(colored(f"Missing required environment variables: {missing_vars}", "red"))
        sys.exit(1) # Exit the application

    # In-memory checkpointer
    thread = {'configurable': {"thread_id": "1"}}

    # create the main graph
    graph = build_main_graph()
    
    # Initialise START_TIME for the performance logs
    init_timer()

    # Initial run
    result = graph.invoke({"problem": initial_input, "max_steps": 3}, config=thread)
    
    # Keep processing interruptions until "No feedback" is input by the user
    while result.get("__interrupt__", ""):

        # Get the interrupt message
        interrupt_message = result["__interrupt__"][0].value
        print(interrupt_message)
        
        # Get user input
        user_input = input("Provide your response or type " + colored("No feedback", "yellow") + " if you approve the plan" + "\n> ")
      
        # Resume and get new result
        result = graph.invoke(Command(resume=user_input), config=thread)

    # Prepare render to Markdown
    custom_theme = Theme({
    "markdown.h1": "bold yellow",
    "markdown.h2": "bold yellow"
    })

    console = Console(theme=custom_theme)
    md = Markdown(result["final_plan"])
    console.print(md)     


if __name__ == "__main__":

    print(colored("\n*** Welcome to the Wellbeing Assistant demo! ***\n", "cyan"))

    default_input = "I'm feeling very stressed at work, because I don't like being surrounded by many people in an open office."
    print('Describe your wellbeing challenge or press', colored('[enter]', "yellow"), 'for', colored('default demo input.', 'yellow'), f'\nDefault input:', colored(f'"{default_input}"\n', "yellow"))
    user_input = input("> ")
    
    if not user_input.strip():
        user_input = default_input

    main(user_input)