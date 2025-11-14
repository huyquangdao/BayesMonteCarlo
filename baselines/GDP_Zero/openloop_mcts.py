import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


class MCTS():
    def __init__(self, game, player, configs) -> None:
        """
        Class vanilla monte carlo tree search
        The code is borrow from the official implementation of GDP-zero
        @param game: the enviorment
        @param player: the dialogue agent.
        @param configs: hyper parameters
        """
        self.game = game
        self.player = player
        self.configs = configs
        # U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
        self.Ns: dict = {}  # saves compute
        self.Nsa: dict = {}
        self.Q: dict = {}
        self.P: dict = {}
        # utility
        self.valid_moves: dict = {}
        self.terminals: dict = {}
        # debugging / more information
        self.Vs: dict = {}
        return

    def _to_string_rep(self, state):
        # for tree search, keep all dialog turns
        return state.to_string_rep(keep_sys_da=True, keep_user_da=True, max_turn_to_display=-1)

    def _init_node(self, state):
        hashable_state = self._to_string_rep(state)
        allowed_actions = self.player.get_valid_moves(state)
        self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
        self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}

        prior, v = self.player.predict(state)
        self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
        self.P[hashable_state] = prior * allowed_actions
        # renormalize
        if np.sum(self.P[hashable_state]) == 0:
            self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
            logger.warning("This should never happen")
        else:
            self.P[hashable_state] /= np.sum(self.P[hashable_state])
        return v

    def search(self, state):
        hashable_state = self._to_string_rep(state)

        is_leaf_node = False
        v = 0.0
        if hashable_state not in self.terminals:
            # selected leaf node, expand
            self.terminals[hashable_state] = self.game.get_dialog_ended(state)
            v = self._init_node(state)
            is_leaf_node = True
        # if this leaf node is terminal, return the value
        if self.terminals[hashable_state] > 0:
            # terminal node
            logger.debug("ended")
            return self.terminals[hashable_state]
        # otherwise, return v
        if is_leaf_node:
            return v

        # existing, continue selection
        # go next state by picking best according to U(s,a)
        best_uct = -float('inf')
        best_action = -1
        for a in self.valid_moves[hashable_state]:
            Ns = self.Ns[hashable_state]
            if Ns == 0:
                Ns = 1e-8
            uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (
                    1 + self.Nsa[hashable_state][a])
            if uct > best_uct:
                best_uct = uct
                best_action = a
        # transition
        next_state = self.game.get_next_state(state, best_action)

        # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
        # 2. if leaf, we will expand it and return the value for backpropagation
        v = self.search(next_state)

        # update stats
        # add in new estimate and average
        self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][
            best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
        self.Ns[hashable_state] += 1
        self.Nsa[hashable_state][best_action] += 1

        # now we are single player, hence just v instead of -v
        return v

    def get_action_prob(self, state):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.Ns:
            # selected leaf node, expand
            logging.warn("querying a state that has not been visited")
            self._init_node(state)
        # get the counts for all moves
        # convert to prob
        prob = np.zeros(self.player.get_valid_moves(state).shape)
        for a in self.valid_moves[hashable_state]:
            prob[a] = self.Nsa[hashable_state][a]
        prob /= prob.sum()
        return prob


class OpenLoopMCTS(MCTS):
    def __init__(self, game, player, configs) -> None:
        """
        Class Open-loop MCTS, the code is borrowed from the GDP-zero model.
        @param game: the enviroment
        @param player: the dialogue agent
        @param configs: hyperparaemeters.
        """
        super().__init__(game, player, configs)
        self.realizations: dict = {}  # state -> list of real DialogSessions
        self.realizations_Vs: dict = {}  # state -> {realization: V(realization)}
        self.realizations_Ns: dict = {}  # state -> {realization: N(realization)}
        self.max_realizations = configs.max_realizations
        return

    def _to_string_rep(self, state):
        """
        Function that converts state to string
        For open loop MCTS, we only consider the system dialogue actions
        @param state: the current state of the conversation.
        @return: a string that represent the dialogue actions of the system.
        """
        # for tree search, keep all dialog turns
        das = []
        for goal in state['pre_goals']:
            da = f"{goal}"
            das.append(da)
        return "__".join(das)

    def _init_node(self, state):
        # convert the current state to a string format
        hashable_state = self._to_string_rep(state)

        # checking actions
        allowed_actions = self.player.get_valid_moves(state)
        self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

        # initialize the Q,V tables.
        self.Ns[hashable_state] = 0
        self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
        self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
        self.realizations[hashable_state] = [state.copy()]

        prior, v = self.player.predict(state)
        # self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
        self.P[hashable_state] = prior * allowed_actions

        # renormalize
        if np.sum(self.P[hashable_state]) == 0:
            raise Exception("This should never happen")
        else:
            self.P[hashable_state] /= np.sum(self.P[hashable_state])
        return v

    def _sample_realization(self, hashable_state):
        rand_i = np.random.randint(len(self.realizations[hashable_state]))
        return self.realizations[hashable_state][rand_i]

    def _add_new_realizations(self, state):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.realizations:
            self.realizations[hashable_state] = []
        if state in self.realizations[hashable_state]:
            return

        self.realizations[hashable_state].append(state.copy())
        if len(self.realizations[hashable_state]) > self.max_realizations:
            # should never happen
            logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
            self.realizations[hashable_state].pop(0)
        return

    def _get_next_state(self, state, best_action):
        goal = self.player.id2goal[best_action]
        prefetch_state = self._to_string_rep(state) + "__" + f"{goal}"
        if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
            # use the cached realization
            return self._sample_realization(prefetch_state)

        # otherwise, generate a new realization
        next_state = self.game.get_next_state(state, self.player.id2goal[best_action])
        return next_state

    def _update_realizations_Vs(self, state, v: float):
        hashable_state = self._to_string_rep(state)
        if hashable_state not in self.realizations_Vs:
            self.realizations_Vs[hashable_state] = {}
            self.realizations_Ns[hashable_state] = {}

        sys_utt = state['dialogue_context'][-2]['content']

        if sys_utt not in self.realizations_Vs[hashable_state]:
            self.realizations_Vs[hashable_state][sys_utt] = 0
            self.realizations_Ns[hashable_state][sys_utt] = 0
        # update
        self.realizations_Ns[hashable_state][sys_utt] += 1
        self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / \
                                                         self.realizations_Ns[hashable_state][sys_utt]
        return

    def search(self, state):
        hashable_state = self._to_string_rep(state)

        # check everytime since state is stochastic, does not map to hashable_state
        terminated_v = self.game.get_dialog_ended(state)

        # check if it is terminal node
        # failed or successfully recommending the target item.
        if terminated_v in [-1,1]:
            logger.debug("ended")
            return terminated_v

        # otherwise, if is nontermial leaf node, we initialize and return v
        if hashable_state not in self.P:
            # selected leaf node, expand it
            # first visit V because v is only evaluated once for a hashable_state
            v = self._init_node(state)
            return v
        else:
            # add only when it is new
            self._add_new_realizations(state)

        # existing, continue selection
        # go next state by picking best according to U(s,a)
        best_uct = -float('inf')
        best_action = -1
        for a in self.valid_moves[hashable_state]:
            Ns = self.Ns[hashable_state]
            if Ns == 0:
                Ns = 1e-8

            # a variant of PUCT
            uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (
                    1 + self.Nsa[hashable_state][a])

            if uct > best_uct:
                best_uct = uct
                best_action = a
        # transition. For open loop, first sample from an existing realization
        state = self._sample_realization(hashable_state)
        next_state = self._get_next_state(state, best_action)

        # 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
        # 2. if leaf, we will expand it and return the value for backpropagation
        v = self.search(next_state)

        # print("score: ", v)
        # print("best action: ", best_action)
        # print("hashable state: ", hashable_state)

        # update stats
        # add in new estimate and average
        self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][
            best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
        self.Ns[hashable_state] += 1
        self.Nsa[hashable_state][best_action] += 1

        # update v to realizations for NLG at inference
        self._update_realizations_Vs(next_state, v)
        # now we are single player, hence just v instead of -v
        return v

    def get_best_realization(self, state, action: int):
        prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[action]
        if prefetch_state not in self.realizations_Vs:
            raise Exception("querying a state that has no realizations sampled before")
        # get the counts for all moves
        # convert to prob
        curr_best_v = -float('inf')
        curr_best_realization = None
        for sys_utt, v in self.realizations_Vs[prefetch_state].items():
            if v > curr_best_v:
                curr_best_v = v
                curr_best_realization = sys_utt
        return curr_best_realization
