import json
import copy
from tqdm import tqdm
from itertools import product
from base.dataset import NegotiationDataset
from config.constants import NEGOTIATION_GOAL2DESCRIPTION, NEGOTIATION_REWRITE_PROMPT, NEGOTIATION_REWRITE_COT_PROMPT, CHATGPT
from utils.prompt import call_llm


class CraiglistBargain(NegotiationDataset):

    def __init__(self, dataset_config, **kwargs):
        """
        constructor for class Cariglist Bargain dataset
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

    def construct_action_mapping(self, combine=True, bin_size=5):
        """
        method that construct the action mapping dictionary for the negotiation dataset
        :param combine:
        :param bin_size:
        :return:
        """
        if combine:
            goal2id = product(self.goals, list(range(bin_size)))
            goal2id = {k: v for v, k in enumerate(goal2id)}
        else:
            goal2id = {k: v for v, k in enumerate(self.goals)}
        return goal2id

    def process_data(self, data):
        """
        method that process the dataset
        :param data: the loaded data
        :return: pre-processed data instances
        """
        all_instances = []
        for conv_id, line in tqdm(enumerate(data)):
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
        task_background = {
            "item_name": conv['item_name'],
            "buyer_price": conv['buyer_price'],
            "buyer_item_description": conv['buyer_item_description'],
            "seller_price": conv['seller_price'],
            "seller_item_description": conv['seller_item_description']
        }
        utts = []
        goals = ["None"]

        for i, utt in enumerate(conv['dialog']):
            is_last_turn = False
            # terminated conversation
            # user turn
            # in the bargain, the user is the seller
            # the system is the buyer
            if utt['speaker'] == 'usr':
                utts.append({'role': 'user', 'content': utt['text']})
            # the system turn
            elif utt['speaker'] == 'sys':
                goal = utt['strategy']
                self.goals.append(goal)

                # if the last utt of the conversation
                # if the last sentence is system response
                if i == len(conv['dialog']) - 1:
                    # if the last utt is system response
                    # no more transition
                    # we do not consider this instance
                    break
                # the second last turn
                elif i == len(conv['dialog']) - 2:
                    done = 1
                    user_res = conv['dialog'][i + 1]['text']
                    is_last_turn = True
                    # else if the last utt i
                # the third last turn
                elif conv['dialog'][-1]['speaker'] == 'sys' and i == len(conv['dialog']) - 3:
                    done = 1
                    user_res = conv['dialog'][i + 1]['text']
                    is_last_turn = True
                # otherwise
                else:
                    done = 0
                    user_res = conv['dialog'][i + 1]['text']
                
                # constructing an instance.
                instance = {
                    "conv_id": conv_id,
                    "response": utt['text'],
                    "goal": goal,
                    "pre_goals": copy.deepcopy(goals),
                    "dialogue_context": copy.deepcopy(utts),
                    "task_background": copy.deepcopy(task_background),
                    "done": done,
                    "usr_response": user_res
                }
                
                # # rewrite the goal description
                # initial_goal_description = NEGOTIATION_GOAL2DESCRIPTION[goal]
                # rewrite_goal_description = self.rewrite_goal_description(
                #     dialogue_context=instance['dialogue_context'],
                #     response=instance['response'],
                #     initial_goal_description=initial_goal_description
                # )

                if len(utts) > 0:
                    instances.append(instance)
                    if is_last_turn:
                        break

                # update the dialogue context
                utts.append({'role': 'assistant', 'content': utt['text']})
                # update the goal path
                goals.append(goal)
                
        # for instance in instances:
        #     print(instance['dialogue_context'])
        #     print(instance['goal'])
        #     print(instance['usr_response'])
        #     print(instance['response'])
        #     print("-" * 50)
        # print(conv['dialog'])
        # assert 1 == 0

        return instances

    def update_instances(self, new_train_instances):
        self.train_instances.extend(new_train_instances)
    
    def rewrite_goal_description(self, dialogue_context, response, initial_goal_description):
        dialogue = ''
        for utt in dialogue_context:
            dialogue += f"{utt['role']}: {utt['content']} "
            
        prompt = [
            {"role": "system", "content": NEGOTIATION_REWRITE_PROMPT},
            {"role": "user", "content": NEGOTIATION_REWRITE_COT_PROMPT.format(dialogue, response, initial_goal_description)}
        ]
                
        # calling the llm to predict the action
        responses = call_llm(prompt, temperature=0.001, max_token= 50,
                             model_type=CHATGPT)
        
        print(initial_goal_description)
        print(responses)
        # assert 1 == 0
        # print(action)
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
        
