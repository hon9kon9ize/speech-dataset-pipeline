import math
import kenlm
import torch
from transformers import LogitsProcessor
from functools import lru_cache
from typing import Union, List, Tuple


LOG_BASE_CHANGE_FACTOR = 1.0 / math.log10(math.e)


class NgramLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        lm_model: Union[str, "kenlm.Model"],
        lm_alpha: float = 0.5,
        lm_beta: float = 0.5,
        lm_start_token_id: int = 50364,
        top_k: int = 50,
    ):
        self.lm: "kenlm.Model" = (
            kenlm.Model(lm_model) if type(lm_model) == str else lm_model
        )
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.lm_start_token_id = lm_start_token_id
        self.top_k = top_k

    @lru_cache(maxsize=None)
    def _score(self, token_ids: Union[List[int], Tuple[int]]):
        next_state = kenlm.State()
        curr_state = kenlm.State()
        prob = 0.0

        if len(token_ids) > 1:
            prob, curr_state = self._score(token_ids[:-1])
        else:
            self.lm.BeginSentenceWrite(curr_state)

        prob += self.lm.BaseScore(curr_state, chr(token_ids[-1] + 100), next_state)
        curr_state, next_state = next_state, curr_state

        return prob, curr_state

    def clear_score_cache(self):
        self._score.cache_clear()

    def __call__(
        self,
        input_ids: torch.LongTensor,  # (beam_size, seq_len)
        scores: torch.FloatTensor,  # (beam_size, vocab_size)
    ) -> torch.FloatTensor:
        n_beams, n_vocab = scores.shape
        top_k_tokens = torch.topk(scores, self.top_k, dim=1).indices

        for i in range(n_beams):
            prefix = input_ids[i].tolist()

            if self.lm_start_token_id not in prefix:
                continue

            # skip first special tokens
            offset_token_index = prefix.index(self.lm_start_token_id) + 1

            if len(prefix) > offset_token_index:
                lm_score = torch.zeros(n_vocab)

                # save last state so that we do not have to recompute the whole sentence
                prob, last_state = self._score(tuple(prefix[offset_token_index:]))

                # calculate all log10 probabilities of all tokens
                # this is not efficient: https://github.com/kpu/kenlm/issues/367

                for k in top_k_tokens[i]:
                    new_token_state = kenlm.State()
                    new_token_score = self.lm.BaseScore(
                        last_state, chr(k.item() + 100), new_token_state
                    )
                    lm_score[k] = new_token_score

                lm_score = torch.FloatTensor(lm_score).to(scores.device)
                lm_score = self.lm_alpha * lm_score * LOG_BASE_CHANGE_FACTOR + 0.5

                scores[i] = scores[i] + lm_score

        return scores
