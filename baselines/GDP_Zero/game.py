import re
import logging

logger = logging.getLogger(__name__)

from baselines.GDP_Zero.utils import update_state_for_open_loop_mcts


class DialogGame(object):
    def __init__(self, game, generation_method, user_simulator):
        """
        constructor for class dialogue game used for the GDP zero model
        :param game: the instantiation of the game
        :param generation_method: the generation method
        :param user_simulator: the user simulator used for generating the user's response
        """
        self.generation_method = generation_method
        self.user_simulator = user_simulator

        # create a game
        self.game = game

    def get_game_ontology(self) -> dict:
        """returns game related information such as dialog acts, slots, etc.
        """
        raise NotImplementedError

    def get_next_state(self, state, action):
        """
        state transition function
        @param state: the current state
        @param action: the predicted action
        @return: the new state.
        """

        state['pred_goal'] = action

        # generate the system response
        system_response = self.generation_method.generate_response(state)

        # generate user response with LLM
        user_response = self.user_simulator.respond(state)

        # update the next state
        next_state = update_state_for_open_loop_mcts(state=state,
                                                     action=action,
                                                     system_response=system_response,
                                                     user_response=user_response)
        return next_state

    def display(self, state):
        string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
        print(string_rep)
        return

    def get_dialog_ended(self, state, reward=3) -> float:
        """returns 0 if not ended, then (in general) 1 if system success, -1 if failure
        """
        # if we're at the beginning of the conversation
        # then the conversation is on going
        if len(state['pre_goals']) == 1:
            return 0.0

        # the conversation length exceeds a predefined threshold
        # then the conversation is failed.
        if len(state['pre_goals']) > self.game.game_config.max_horizon:
            return -1

        # otherwise we compute llm-based assessment
        _, done, _ = self.game.compute_reward(state,
                                                   state['pred_goal'],
                                                   state['dialogue_context'][-2]['content'],
                                                   None
                                                   )
        # no intermediate reward.
        return done
