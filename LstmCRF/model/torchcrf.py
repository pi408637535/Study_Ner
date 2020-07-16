#coding:utf-8
__version__ = '0.7.2'

from typing import List, Optional

import torch
import torch.nn as nn
import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchtext
import random
import copy

class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, tag2id, num_tags: int, device, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
    
        self.device = device
        self.transitions = nn.Parameter(
            torch.randn(num_tags, num_tags))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        #START_TAG, STOP_TAG在data_utils添加过
        START_TAG = "<start>"
        STOP_TAG = "<stop>"
        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000
        self.tagset_size = num_tags
        self.tag_to_ix = tag2id

        #self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'token_mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        loss = forward_score - gold_score

        if reduction == 'none':
            return loss
        if reduction == 'sum':
            return loss.sum()
        if reduction == 'mean':
            return loss.mean()
        assert reduction == 'token_mean'

        return loss.sum() / mask.float().sum()

    def _viterbi_decode_old(self, feats, mask):
        seq_length, tag_size = feats.size()
        f = torch.zeros(seq_length, tag_size)

        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix["<start>"]] = 0

        forward_var = init_vvars

        pi = [[-1 for j in range(tag_size)] for i in range(seq_length)]

        for i, feat in enumerate(feats):
            viterbi_var = []

            for tag in range(tag_size):
                next_tag = forward_var + self.transitions[tag]
                best_tag_id = next_tag.argmax(dim=1)

                viterbi_var.append(next_tag[0][best_tag_id])
                pi[i][tag] = best_tag_id.numpy()[0]
            forward_var = (t.cat(viterbi_var, dim=0) + feat).view(1, -1)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix["<stop>"]]
        best_tag_id = terminal_var.argmax(dim=1)

        path = [best_tag_id.numpy()[0]]
        x = seq_length - 1
        y = best_tag_id

        for k in range(1, seq_length):
            path.append(pi[x][y])  # STOP_TAG has been add so I lift this one
            y = pi[x][y]
            x -= 1

        data = [self.idx2tag[ele] for ele in path[::-1]]
        print(data)

    def _viterbi_decode(self, feats, mask):
        """
            :param feats: (seq_len, batch_size, tag_size)
            :param mask: (seq_len, batch_size)
            :return best_path: (seq_len, batch_size)
            """
        seq_len, batch_size, tag_size = feats.size()
        # initialize scores in log space
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, self.tag_to_ix["<start>"]] = 0

        scores = scores.to(self.device)

        pointers = []

        mask = mask.float()

        # forward
        for t, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            scores_t = scores.unsqueeze(1) + self.transitions.unsqueeze(0)  # (batch_size, tag_size, tag_size)
            # max along current_tag to obtain: next_tag score, current_tag pointer
            scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
            scores_t += feat
            pointers.append(pointer)
            mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_t * mask_t + scores * (1 - mask_t)
        pointers = torch.stack(pointers, 0)  # (seq_len, batch_size, tag_size)

        scores += self.transitions[self.tag_to_ix["<stop>"]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
        # backtracking
        best_path = best_tag.unsqueeze(-1).tolist()  # list shape (batch_size, 1)
        for i in range(batch_size):
            best_tag_i = best_tag[i]
            seq_len_i = int(mask[:, i].sum())
            for ptr_t in reversed(pointers[:seq_len_i, i]):
                # ptr_t shape (tag_size, )
                best_tag_i = ptr_t[best_tag_i].item()
                best_path[i].append(best_tag_i)
            # pop first tag
            best_path[i].pop()
            # reverse order
            best_path[i].reverse()
        return best_path

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)


        return self._viterbi_decode(emissions, mask)

    def _forward_alg(self, feats, mask):
        """
        Arg:
          feats: (seq_len, batch_size, tag_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        # initialize alpha to zero in log space
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        # alpha in START_TAG is 1
        alpha[:, self.tag_to_ix["<start>"]] = 0

        alpha = alpha.to(self.device)
        mask = mask.float()

        for i, feat in enumerate(feats):
            # broadcast dimension: (batch_size, next_tag, current_tag)
            # emit_score is the same regardless of current_tag, so we broadcast along current_tag
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1)
            # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
            transition_score = self.transitions.unsqueeze(0)  # (1, tag_size, tag_size)
            # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size)
            alpha_score = alpha_score + transition_score + emit_score
            # log_sum_exp along current_tag dimension to get next_tag alpha
            mask_t = mask[i].unsqueeze(-1)
            alpha = t.logsumexp(alpha_score, -1) * mask_t + alpha * (1 - mask_t)  # (batch_size, tag_size)
        # arrive at END_TAG
        alpha = alpha + self.transitions[self.tag_to_ix["<stop>"]].unsqueeze(0)

        return t.logsumexp(alpha, -1)

    #计算所有路径
    #feats已经被transpose. feats:seq,batch,tags
    def _forward_alg_old(self, feats, mask):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix["<start>"]] = 0.
        batch = feats.shape[1]


        previous = init_alphas.expand(batch, -1) #previous batch,tag
        previous = previous.to(self.device)
        mask = mask.float()

        for i,feat in enumerate(feats): #feats:seq,batch,tag     feat: 1,batch,tag
            word_tag_score = [] # 计算逐个word->tag score
            for tag in range(self.tagset_size):
                word_emission = feat[:,tag].unsqueeze(dim=1).expand(-1,54) #word_emission:batch,tags
                word_tag_transition = self.transitions[tag, :].unsqueeze(dim=0).expand(batch, -1)   #word_tag_transition:batch,tags
                word_score = previous + word_emission + word_tag_transition #word_score:batch,tags

                word_score = t.logsumexp(word_score, dim=1) #word_score:batch,1

                word_score = word_score * mask[i]


                word_tag_score.append(word_score)

            previous = t.cat(word_tag_score, dim=0).view(batch, -1)

        terminal_var = previous + self.transitions[self.tag_to_ix["<stop>"]].expand(batch, -1)
        all_path_score = t.logsumexp(terminal_var, dim=1)
        return all_path_score

    def _forward_alg2(self, feats, mask):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix["<start>"]] = 0.
        batch = feats.shape[1]


        previous = init_alphas.expand(batch, -1) #previous batch,tag
        previous = previous.to(self.device)
        mask = mask.float()

        for i,feat in enumerate(feats): #feats:seq,batch,tag     feat: 1,batch,tag
            word_tag_score = [] # 计算逐个word->tag score
            for tag in range(self.tagset_size):
                word_emission = feat[:,tag].unsqueeze(dim=1).expand(-1,54) #word_emission:batch,tags
                word_tag_transition = self.transitions[tag, :].unsqueeze(dim=0).expand(batch, -1)   #word_tag_transition:batch,tags
                word_score = previous + word_emission + word_tag_transition #word_score:batch,tags

                word_score = t.logsumexp(word_score, dim=1) #word_score:batch,1

                word_score = word_score * mask[i]


                word_tag_score.append(word_score)

            previous = t.cat(word_tag_score, dim=0).view(batch, -1)

        terminal_var = previous + self.transitions[self.tag_to_ix["<stop>"]].expand(batch, -1)
        all_path_score = t.logsumexp(terminal_var, dim=1)
        return all_path_score



    def _score_sentence(self, feats, tags, mask):
        #feats: batch,seq,tag
        #tags:tag
        batch = feats.shape[1]
        #feats: seq, batch
        score = torch.zeros(batch).to(self.device)
        tags = tags.permute(0,1)

        #temp = t.tensor([self.tag_to_ix["<start>"]], dtype=torch.long).view(1)
        temp = tags.new_full((1, batch), fill_value=self.tag_to_ix["<start>"])
        temp = temp.to(self.device)

        tags = torch.cat([temp, tags], dim=0)
        #tags = t.transpose(tags, 1, 0 ) #seq,batch
        #tags = tags.expand(batch, 1)
        #tags[0] is start tag.
        mask = mask.float()
        for i,feat in enumerate(feats):
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[i + 1])])
            transition_score = torch.stack([self.transitions[tags[i + 1, b], tags[i, b]] for b in range(batch)])
            score += (emit_score + transition_score) * mask[i]

        transition_to_end = torch.stack([self.transitions[self.tag_to_ix["<stop>"], tag[mask[:, b].sum().long()]] for b, tag in
                                         enumerate(tags.transpose(0, 1))])


        score += transition_to_end
        return score

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

