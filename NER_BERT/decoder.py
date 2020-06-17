import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@torch.jit.script
def viterbi_decode_single_jit(tag_sequence, transition_matrix):
    top_k = 1
    sequence_length, num_tags = tag_sequence.size()
    num_tags = num_tags + 2

    zero_sentinel = torch.zeros(1, num_tags)
    extra_tags_sentinel = torch.ones(sequence_length, 2) * -10000
    tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
    tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)

    path_indices = torch.zeros(num_tags, dtype=torch.long, device=tag_sequence.device).unsqueeze(0)
    path_scores = tag_sequence[0, :].unsqueeze(0)

    for t in range(1, tag_sequence.size(0)):
        summed_potentials = path_scores.unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        scores, paths = torch.topk(summed_potentials, k=top_k, dim=0)

        path_scores = tag_sequence[t, :].unsqueeze(0) + scores
        path_indices = torch.cat([path_indices, paths], 0)

    path_indices = path_indices[1:]
    path_scores_v = path_scores.view(-1)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=top_k, dim=0)

    n_paths_indices = path_indices.size(0)

    viterbi_paths = torch.zeros(sequence_length, dtype=torch.long, device=tag_sequence.device).unsqueeze(0)
    tag_scores = torch.zeros(sequence_length, device=tag_sequence.device).unsqueeze(0)
    for i in range(top_k):
        viterbi_path = best_paths[0].unsqueeze(0)

        for k in range(n_paths_indices):
            t_rev = n_paths_indices - k - 1
            backward_timestep = path_indices[t_rev, :]
            tag_id = torch.index_select(backward_timestep.view(-1), 0, viterbi_path[-1])
            viterbi_path = torch.cat([viterbi_path, tag_id], -1)

        viterbi_path = viterbi_path.flip(0)
        viterbi_path = viterbi_path % num_tags
        viterbi_path = viterbi_path[1:-1]
        viterbi_paths = torch.cat([viterbi_paths, viterbi_path.unsqueeze(0)], 0)

        tag_score = torch.gather(tag_sequence[1:-1], 1, viterbi_path.unsqueeze(-1)).view(-1)
        tag_scores = torch.cat([tag_scores, tag_score.unsqueeze(0)], 0)
    viterbi_paths = viterbi_paths[1:]
    tag_scores = tag_scores[1:]
    return viterbi_paths, tag_scores.exp(),  viterbi_scores.exp()

def predict_ner_single_jit(logits_ner, labels, transition_matrix, tag_pad_id):
    mask = labels.ne(tag_pad_id)
    logits_padded = torch.masked_select(logits_ner, mask.unsqueeze(2)).view(-1, logits_ner.size()[-1])
    return viterbi_decode_single_jit(logits_padded, transition_matrix)


#######
def viterbi_decode_single_python(e, t):
    num_tags = len(e[0])
    seq_len = len(e)
    start_tag_id = num_tags
    end_tag_id = num_tags + 1

    dp_links = []
    dp = [0.] * num_tags
    curr_dp_links = []
    for j in range(num_tags):
        dp[j] = t[start_tag_id, j] + e[0][j]
        curr_dp_links.append(-1)
    dp_links.append(curr_dp_links)

    for i in range(1, seq_len):
        new_dp = []
        curr_dp_links = []
        for j in range(num_tags):
            all_candidates = [np.logaddexp(t[k, j] + e[i][j], dp[k]) for k in range(num_tags)]
            max_k = max(range(num_tags), key=lambda i: all_candidates[i])
            new_dp.append(all_candidates[max_k])
            curr_dp_links.append(max_k)
        dp = new_dp
        dp_links.append(curr_dp_links)

    all_candidates = [np.logaddexp(t[k, end_tag_id], dp[k]) for k in range(num_tags)]
    max_k = max(range(num_tags), key=lambda i: all_candidates[i])
    sentence_score = all_candidates[max_k]

    all_labels = [max_k]
    all_labels_score = [t[max_k, end_tag_id]]

    for i in range(seq_len - 1, 0, -1):
        curr_k = dp_links[i][max_k]
        all_labels.append(curr_k)
        all_labels_score.append(t[curr_k, max_k] + e[i][max_k])
        max_k = curr_k

    return sentence_score, all_labels[::-1], all_labels_score[::-1]

def predict_ner_single_python(logits_ner, labels, transitions, tag_pad_id):
    mask = labels.ne(tag_pad_id)
    logits_ner_padded = torch.masked_select(logits_ner, mask.unsqueeze(2)).view(-1, logits_ner.size()[-1])
    logits_ner_padded_lsf = F.log_softmax(logits_ner_padded, dim=-1)
    score_sentence, pred_tags, score_tags = viterbi_decode_single_python(logits_ner_padded_lsf.cpu().data.numpy(), transitions.cpu().data.numpy())
    return [pred_tags], [np.exp(score_tags)], [np.exp(score_sentence)]
