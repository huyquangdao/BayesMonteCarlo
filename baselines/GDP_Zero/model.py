import numpy as np

from tqdm import tqdm

from base.model import Model
from baselines.GDP_Zero.openloop_mcts import OpenLoopMCTS
from baselines.GDP_Zero.player import LLMPlayer
from baselines.GDP_Zero.game import DialogGame


class GDPZeroModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for ProCOT dialogue policy
        :param model_config: the configuration of the model
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        self.model_config = model_config
        self.temperature = self.model_config.temperature
        self.max_gen_tokens = self.model_config.max_gen_tokens

        # the prompt for the player
        self.prompt = self.model_config.prompt

    def set_agent(self, game, generation_method, user_simulator, action_mapping):
        """
        function that set the mcts agent
        :param game: the instantiation of the current game
        :return: None
        """
        # the player and the mcts agent for the GDP-Zero model
        player = LLMPlayer(game.game_config, action_mapping, self.model_config)

        # initializing the game
        mcts_game = DialogGame(game, generation_method, user_simulator)

        # initializing the mtcs agent
        self.mcts_agent = OpenLoopMCTS(mcts_game, player, self.model_config)

    def forward(self, inputs, game, generation_method, user_simulator, action_mapping):
        """
        method that predicts the action using the Standard Prompting model
        :param inputs: the input of the model
        :return:
        """
        # re-init the mcts algorithm at each step
        self.set_agent(game, generation_method, user_simulator, action_mapping)

        # predict system action using open-loop monte-carlo tree search
        for _ in tqdm(range(self.model_config.rollouts)):
            self.mcts_agent.search(inputs)

        # processing to get the mcts policy
        mcts_policy = self.mcts_agent.get_action_prob(inputs)

        # get the dialogue action
        action = self.mcts_agent.player.id2goal[np.argmax(mcts_policy)]
        return action
