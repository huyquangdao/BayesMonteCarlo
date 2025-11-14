import json
from tqdm import tqdm
import copy
from base.dataset import PersuationDataset
from config.constants import P4G_GOALS, CHATGPT, LLAMA3
from utils.prompt import call_llm


class Persuation4Good(PersuationDataset):
    
    def __init__(self, dataset_config, **kwargs):
        """
        constructor for class Persuation4Good dataset
        :param dataset_config: the configuration of the dataset
        :param kwargs: other keywords parameters
        """
        super().__init__(dataset_config, **kwargs)

    def read_data(self, data_path):
        """
        method that load the data from a file path
        :param data_path: the path to the dataset
        :return: a list of raw data
        """
        with open(data_path, 'r') as f:
            data = f.readlines()
            assert len(data) > 0
            return data

    def repurpose_dataset(self, data):
        """
        method that repurpose the dataset for the negotiation scenario
        :param data: the loaded raw data
        :return: a list of re-purposed data
        """
        new_data = []
        # for negotiation, there is no need for re-purposing the dataset
        # we employ a simple "processing" step here to keep the pipeline consistent
        for line in data:
            line = json.loads(line)
            new_data.append(line)
        return new_data

    def process_data(self, data):
        """
        method that process the dataset
        :param data: the loaded data
        :return: pre-processed data instances
        """
        all_instances = []
        for conv_id, line in enumerate(tqdm(data)):
            instances = self.construct_instances(conv_id, line)
            all_instances.extend(instances)
        return all_instances

    def construct_instances(self, conv_id, conv):
        """
        method that processes the data to obtain a list of instances
        :param conv_id: the id of the conversation
        :param conv: the conversation data
        :return: a list of instances.
        """
        instances = []
        
        # no task background
        task_background = {}
        utts = []
        goals = ["None"]
        
        # for turn in conv['dialog']:
        #     print(turn)
        # assert 1 == 0
        
        assert len(conv['dialog']) == len(conv['label'])
        for i, (turn, act) in enumerate(list(zip(conv['dialog'], conv['label']))):
            is_last_turn = False
            # user turn
            # in the bargain, the user is the seller
            # the system is the buyer
            
            # skip the first turn 
            # can be used as the seed for online evaluation           
            if i == 0:
                for role in turn.keys():
                    text = ""
                    for sent in turn[role]:
                        text += sent + ". "
                    role = 'assistant' if role == 'er' else 'user'
                    utts.append({'role': role, 'content': text})
                    
                    # update the previous goals
                    for a in act['er']:
                        if a in P4G_GOALS:
                            goals.append(a)            
            # ee -> er       
            else:     
                for role in turn.keys():
                    # user turn
                    if role == 'ee':
                        text = ""
                        for sent in turn[role]:
                            text += sent + ". "
                        utts.append({'role': 'user', 'content': text})
                        
                    # the system turn
                    # construct instances
                    elif role == 'er':
                        # consider the case that a single turn has multiple actions
                        assert len(turn[role]) == len(act[role])
                        for utt, a in list(zip(turn[role], act[role])):
                            if a not in P4G_GOALS:
                                # update the dialogue context
                                utts.append({'role': 'assistant', 'content': utt})
                                continue
                            
                            goal = a
                            self.goals.append(goal)
                            
                            # constructing an instance.
                            instance = {
                                "conv_id": conv_id,
                                "response": utt,
                                "goal": goal,
                                "pre_goals": copy.deepcopy(goals),
                                "dialogue_context": copy.deepcopy(utts),
                                "task_background": copy.deepcopy(task_background),
                            }
                            
                            if len(utts) > 0:
                                instances.append(instance)

                            # instances.append(instance)
                            # update the dialogue context
                            utts.append({'role': 'assistant', 'content': utt})
                            
                            # update the goal path
                            goals.append(goal)
        
        return instances

    
    