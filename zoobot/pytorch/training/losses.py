from typing import Tuple
import logging

import torch
# import pyro


class CustomMultiQuestionLoss(torch.nn.Module):
    def __init__(self, question_answer_pairs: dict, question_functional_loss, careful=False, sum_over_questions=False):
        """Custom loss function for multi-question tasks.

        Args:
            question_answer_pairs (dict): Mapping of questions to their answer keys. Each key is a question 
                string and each value is a list of answer strings for that question.
            question_functional_loss (callable): Functional loss to apply to each question. Should accept 
                predictions and targets and return a loss tensor.
            careful (bool, optional): **Deprecated**. No longer needed. Defaults to False.
            sum_over_questions (bool, optional): If True, sum the losses over questions to return a single 
                loss value per galaxy. If False, return a loss value per question. Defaults to False.
        """
        super().__init__()
        self.question_answer_pairs = question_answer_pairs
        self.question_functional_loss = question_functional_loss
        # looks similar to the RegressionBaseline init, but this is a different self
        self.answer_keys = [q + a for q, a_list in self.question_answer_pairs.items() for a in a_list]
        logging.info(f'answer keys: {self.answer_keys}')

        self.careful = careful
        assert not self.careful, "careful mode is deprecated, no longer needed"
        self.sum_over_questions = sum_over_questions


    def forward(self, inputs: torch.Tensor, targets: dict) -> torch.Tensor:
        """Compute the loss for multi-question predictions.

        Args:
            inputs (torch.Tensor): Prediction vector of shape (batch_size, num_answer_keys). 
                Contains predicted fractions for all answer keys across all questions.
            targets (dict): Dictionary with answer keys as keys and target values as values. 
                Each value has shape (batch_size,) containing the target counts/votes.

        Returns:
            torch.Tensor: Loss tensor. If sum_over_questions is True, returns shape (batch_size,) 
                with one loss value per galaxy. If False, returns shape (batch_size, num_questions) 
                with one loss value per question per galaxy.
        """
        # inputs, prediction vector, is B x N, where N is the number of answer keys (fractions). Might change to dictlike.
        # targets, labels from datamodule, is dictlike with keys of answer_keys, each with values of shape (B)

        # deprecated now, lot of looping for a dict and nans seem rare
        # if self.careful:
        #     # some models give occasional nans for all predictions on a specific galaxy/row
        #     # these are inputs to the loss and only happen many epochs in so probably not a case of bad labels, but rather some instability during training
        #     # handle this by setting loss=0 for those rows and throwing a warning
        #     nan_prediction = torch.isnan(inputs) | torch.isinf(inputs)
        #     if nan_prediction.any():
        #         logging.warning(f'Found nan values in inputs: {inputs}')
        #     safety_value = torch.ones(1, device=inputs.device, dtype=inputs.dtype)  # fill with 1 everywhere i.e. fully uncertain
        #     inputs = torch.where(condition=nan_prediction, input=safety_value, other=inputs)

        q_losses = []

        # logging.info(targets.keys())  # should be answer_keys (and answer_fraction_keys, but ignored here)
                
        for question, answers in self.question_answer_pairs.items():

            q_answer_keys = [question + answer for answer in answers]

            targets_for_answers = torch.stack([targets[key] for key in q_answer_keys], dim=1).int()

            # work out the answer indices for the prediction vector
            answer_indices = [self.answer_keys.index(key) for key in q_answer_keys]
            predictions_for_answers = inputs[:, answer_indices]
            # now these are two vectors and ready for the functional loss
            
            question_loss = self.question_functional_loss(predictions_for_answers, targets_for_answers)
            q_losses.append(question_loss)

        total_loss = torch.stack(q_losses, dim=1)

 

        if self.sum_over_questions:
            # additionally sum over questions, i.e. return a single loss value per galaxy
            return torch.sum(total_loss, dim=1)
        else:
            # return a loss value per question, i.e. shape (num_galaxies, num_questions)
            return total_loss


# inputs, targets format
def get_dirichlet_neg_log_prob(concentrations_for_q, labels_for_q) -> torch.Tensor:
    """
    Negative log likelihood of ``labels_for_q`` being drawn from Dirichlet-Multinomial distribution with ``concentrations_for_q`` concentrations.
    This loss is for one question. Sum over multiple questions if needed (assumes independent).
    Applied by :class:`CustomMultiQuestionLoss`, above, if passed as the `question_functional_loss` argument.

    Args:
        labels_for_q (torch.Tensor): observed labels (count of volunteer responses) of shape (batch, answer)
        concentrations_for_q (torch.Tensor): concentrations for multinomial-dirichlet predicting the observed labels, of shape (batch, answer)

    Returns:
        torch.Tensor: negative log. prob per galaxy, of shape (batch_dim).
    """
    # https://docs.pyro.ai/en/stable/distributions.html#dirichletmultinomial
    # .int()s avoid rounding errors causing loss of around 1e-5 for questions with 0 votes
    # dist = pyro.distributions.DirichletMultinomial(
    #     total_count=total_count.int(), 
    #     concentration=concentrations_for_q, 
    #     is_sparse=False, validate_args=True
    # )
    # dirichlet_neg_log_prob = -dist.log_prob(labels_for_q.int())  # important minus sign
    # print(f"concentrations_for_q: {concentrations_for_q.shape}, labels_for_q: {labels_for_q.shape}")


    # manual equivalent, removes pyro dependency
    dirichlet_neg_log_prob = -log_prob(concentrations_for_q, labels_for_q.int())

    return dirichlet_neg_log_prob  # shape (batch_dim,)

# https://docs.pyro.ai/en/stable/_modules/pyro/distributions/conjugate.html#DirichletMultinomial

def log_prob(alpha, value) -> torch.Tensor:
    """Compute log probability for Dirichlet-Multinomial distribution.
    
    Manual implementation equivalent to pyro.distributions.DirichletMultinomial.log_prob()
    to remove pyro dependency.
    
    Args:
        alpha (torch.Tensor): Concentration parameters of shape (batch, categories).
        value (torch.Tensor): Observed counts of shape (batch, categories).
        
    Returns:
        torch.Tensor: Log probability of shape (batch,).
    """
    return _log_beta_1(alpha.sum(-1), value.sum(-1)) - _log_beta_1(alpha, value).sum(-1)

def _log_beta_1(alpha, value) -> torch.Tensor:
    """Helper function to compute log beta function for Dirichlet-Multinomial.
    
    Args:
        alpha (torch.Tensor): Concentration parameters.
        value (torch.Tensor): Observed counts.
        
    Returns:
        torch.Tensor: Log beta function values.
    """
    return torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)

