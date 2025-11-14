import random
import copy
import numpy as np


def round_nearest(x, a):
    return round(x / a) * a

def randomly_sample_demonstrations(all_convs, instance, k=1):
    """
    function that randomly sample 1 demonstrations from the set of all training conversations.
    here we first filter out a set of examples that have the same target goal with the given one.
    Then we randomly choose 1 demonstration from the candidate set.
    @param all_convs: set of all training conversations
    @param instance: an instance which is a dictionary of dialogue context, task background.
    @param k: the number of sampled demonstrations, default = 1
    @return: a randomly chosen conversation.
    """
    candidate_instances = [x for x in all_convs if
                           x['target_goal'] == instance['task_background']['target_goal']]

    return random.choices(candidate_instances, k=k)


def create_target_set(train_convs, test_instances, num_items=10, shuffle=False, domain = 'movie'):
    """
    function that creates a target item set for the recommendation scenario
    @param train_convs: a list of conversations from training set.
    @param test_instances: a list of test instances.
    @param num_items: the number of target item
    :param: shuffle: True if shuffering the dataset
    @return: a list of dictionary which contain information about the target item (name, goal and demonstration)
    """
    # create the target item set
    all_test_targets = []
    for instance in test_instances:
        all_test_targets.append(instance['task_background']['target_topic'])

    # copy instances before selecting target items
    copied_test_instances = copy.deepcopy(test_instances)

    if shuffle:
        random.shuffle(copied_test_instances)

    # get the set of items from the test set.
    i = 0
    selected_set = []
    selected_set_names = []

    # selecting target items
    while len(selected_set) < num_items and i < len(copied_test_instances):
        instance = copied_test_instances[i]
        current_domain = instance['task_background']['target_goal'].split(" ")[0].strip().lower()
        if domain != "all":
            if instance['task_background']['target_topic'] in selected_set_names or current_domain != domain:
                i += 1
                continue

        # sample a demonstration for user simulator:
        demonstrations = randomly_sample_demonstrations(
            all_convs=train_convs,
            instance=instance
        )

        # create the target
        target = {
            "topic": instance['task_background']['target_topic'],
            "goal": instance['task_background']['target_goal'],
            "topic_set": instance['task_background']['topic_set'],
            "demonstration": demonstrations[0]
        }

        selected_set.append(target)
        selected_set_names.append(target['topic'])
        i += 1
    
    return selected_set


def random_weights(
        dim: int, n: int = 1, dist: str = "dirichlet", seed: int = None,
        rng: np.random.Generator = None, p = 0.01
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.
    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    elif dist == "uniform":
        x = 1 / dim
        w = np.array([[x for t in range(dim)]])
    else:
        raise ValueError(f"Unknown distribution {dist}")
    new_w = []
    for y in w:
        new_w.append([round_nearest(x, p) for x in y])
    if n == 1:
        return new_w[0]
    return new_w


def create_cases(test_instances, num_cases=100, shuffle=False):
    """
    method that create a set of negotiation cases for the negotiation and emotional support scenarios.
    :param test_instances: a list of test instances
    :param num_cases: the number of sampled cases.
    :param shuffle: True if shuffering the dataset.
    :return:
    """
    # get the unique cases
    unique_cases = []
    selected_list = []
    for instance in test_instances:
        conv_id = instance['conv_id']
        if conv_id not in selected_list:
            unique_cases.append(instance)
            selected_list.append(conv_id)
            
    # create the set of cases
    if len(unique_cases) > num_cases:
        all_test_cases = random.sample(unique_cases, num_cases)
    else:
        all_test_cases = unique_cases
    
    # shuffering the negotiation cases.
    if shuffle:
        random.shuffle(all_test_cases)
    return all_test_cases


def save_conversation_for_human_evaluation(conv_path, conv):
    """functions that save the conversations for human evaluation

    Args:
        conv_path (_type_): _description_
        conv (_type_): _description_
    """
    task_background = conv['task_background']
    turns = conv['dialogue_context']
    with open(conv_path, "w") as f:
        item_name = task_background['item_name']
        buyer_price = task_background['buyer_price']
        seller_price = task_background['seller_price']
        f.write(f"[ITEM NAME] : {item_name}" + "\n")
        f.write(f"[BUYER PRICE] : {buyer_price}" + "\n")
        f.write(f"[SELLER PRICE] : {seller_price}" + "\n")
        f.write(f"-"*50 + "\n")
        for turn in turns:
            role = turn['role']
            content = turn['content']
            f.write(f"[{role}] : {content}" + "\n")
        f.write(f"-"*50 + "\n")  
        f.write("Deal Achievement: " + "\n")
        f.write("Negotiation Equity: " + "\n")
        f.write("Buyer's Benifit: " + "\n")
