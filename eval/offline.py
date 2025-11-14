import torch

from base.evaluator import Evaluator


class OfflineEvaluator(Evaluator):

    def __init__(self, metrics, policy_eval=True):
        """
        constructor for class offline evaluation
        :param metrics: set of metric classes
        :param policy_eval: True if it is policy evaluator else false
        """
        super().__init__()
        self.metrics = metrics
        self.preds = []
        self.labels = []
        self.policy_eval = policy_eval
        self.values = {}

    def record(self, preds, labels):
        """
        method that records the values of predictions and labels during evaluation
        :param preds: the predictions of the model
        :param labels: the ground truth labels
        :return: None
        """
        # pre-processing for policy evaluation
        if self.policy_eval:
            # reformat the predtions and labels
            if isinstance(preds, torch.Tensor):
                # predictive policies, e.g BERT, RTCP
                preds = preds.argmax(-1).detach().cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()
            else:
                # generative models, e.g color , unimind or tcp
                preds = [x.split(':')[-1].strip() for x in preds]
                labels = [x.split(':')[-1].strip() for x in labels]

        # generation evaluation
        self.preds.extend(preds)
        self.labels.extend(labels)
        
    def set_eval_mode(self, is_policy_eval):
        self.is_policy_eval = is_policy_eval

    def report(self, step=None):
        """
        method that reports the values of metrics
        :return: None
        """
        # compute the values of metrics
        for metric in self.metrics:
            self.values[metric.__class__.__name__] = metric.compute(self.preds, self.labels)
        return self.values

    def reset(self):
        """
        method that resets values of metrics
        :return: None
        """
        self.preds = []
        self.labels = []
        for metric in self.metrics:
            self.values[str(metric.__class__.__name__)] = 0.0
