import json
from tqdm import tqdm
import copy
from base.dataset import EmotionalSupportDataset
from config.constants import ES_CONV_GOAL2DESCRIPTION, EMOTIONAL_SUPPORT_REWRITE_PROMPT, EMOTIONAL_SUPPORT_REWRITE_COT_PROMPT, CHATGPT, LLAMA3
from utils.prompt import call_llm


class ESConv(EmotionalSupportDataset):

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
        task_background = {
            "emotion_type": conv['emotion_type'],
            "problem_type": conv['problem_type'],
            "situation": conv['situation'],
        }
        utts = []
        goals = ["None"]
        
        for i, utt in enumerate(conv['dialog']):
            is_last_turn = False
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
                }
                
                # # rewrite the goal description
                # initial_goal_description = ES_CONV_GOAL2DESCRIPTION[goal]
                # rewrite_goal_description = self.rewrite_goal_description(
                #     dialogue_context=instance['dialogue_context'],
                #     response=instance['response'],
                #     initial_goal_description=initial_goal_description
                # )

                if len(utts) > 0:
                    instances.append(instance)
                    if is_last_turn:
                        break

                # instances.append(instance)
                # update the dialogue context
                utts.append({'role': 'assistant', 'content': utt['text']})
                # update the goal path
                goals.append(goal)

        return instances

    def rewrite_goal_description(self, dialogue_context, response, initial_goal_description):
        dialogue = ''
        for utt in dialogue_context:
            dialogue += f"{utt['role']}: {utt['content']} "
        prompt = [
            {"role": "system", "content": EMOTIONAL_SUPPORT_REWRITE_PROMPT},
            {"role": "user", "content": EMOTIONAL_SUPPORT_REWRITE_COT_PROMPT.format(dialogue, response, initial_goal_description)}
        ]
                
        # calling the llm to predict the action
        responses = call_llm(prompt, temperature=0.001, max_token= 50,
                             model_type=LLAMA3)
        
        print(initial_goal_description)
        print(responses)
        # assert 1 == 0
        # print(action)
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
        
