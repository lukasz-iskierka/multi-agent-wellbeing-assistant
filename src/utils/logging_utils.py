import time
from datetime import datetime
from termcolor import colored

# Use init_timer() to initialise START_TIME from demo_run.py or wellbeing_assistant.ipynb
START_TIME = None


def init_timer():
    """Initialise START_TIME as a global variable"""
    
    global START_TIME
    START_TIME = time.time()


# Custom function for logs. 
def log(message: str):

    """Print a log together with passed message."""

    # Check if START_TIME was initialised
    if START_TIME is None:
        raise RuntimeError(colored("START_TIME was not initialised —> call init_timer() first.", "red"))
    
    # Record current time
    now = datetime.now().strftime("%H:%M:%S")

    elapsed = time.time() - START_TIME

    if "successfully" in message:
        print(colored(f"[{now} | +{elapsed:06.2f}s]", "light_grey"), colored(f"{message}", "green")) # Print in green
    else:
        print(colored(f"[{now} | +{elapsed:06.2f}s]", "light_grey"), colored(f"{message}", "yellow")) # Print in yellow