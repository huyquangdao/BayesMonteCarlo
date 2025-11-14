import copy
from utils.prompt import call_llm
import numpy as np

def update_state_for_open_loop_mcts(state, system_response, user_response, action):
    """
    function that upate the state for open loop MCTS.
    @param state: the current state of the dialogue
    @param system_response: the system response
    @param user_response: the user response
    @param action: predicted action
    @return: the next state
    """
    new_state = copy.deepcopy(state)
    new_state['dialogue_context'].append(
        {"role": "assistant", "content": system_response}
    )
    new_state['dialogue_context'].append(
        {"role": "user", "content": user_response}
    )
    goal = action
    new_state['pre_goals'].append(goal)
    return new_state



