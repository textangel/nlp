import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM

pre_trained_weights = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pre_trained_weights)


def evaluate_one_word(tokenized_text, topk=4):
    """
    tokenized_text: tokenized_text
    """
    print(f"Number of [MASK]: {tokenized_text.count('[MASK]')}")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_index = [ix for ix, x in enumerate(tokenized_text) if x == "[MASK]"]

    segments_ids = [0] * len(tokenized_text)
    next_sent_start_ix = tokenized_text.index('[SEP]')
    segments_ids[next_sent_start_ix + 1:] = [1] * len(segments_ids[next_sent_start_ix + 1:])

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertForMaskedLM.from_pretrained(pre_trained_weights)
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0][0, masked_index]
        probs_for_mask = F.softmax(predictions, dim=1)

    # confirm we were able to predict the output
    predicted_indices = torch.topk(probs_for_mask, k=topk).indices
    predicted_probs = -1 * torch.topk(probs_for_mask, k=topk).values.numpy()
    predicted_tokens = [tokenizer.convert_ids_to_tokens(tok) for tok in predicted_indices]
    return dict(zip(predicted_probs[0], predicted_tokens[0]))


def get_num_masks(tokenized_text):
    return tokenized_text.count('[MASK]')


def update_probs(prb, new_prb):
    return -1 * prb * new_prb


def stepwise_beam_search(tokenized_text):
    num_masks = get_num_masks(tokenized_text)
    beam_size = 20
    eval_count = beam_size
    best_of_len = {0: [(-1, [])]}
    for length in range(1, num_masks + 1):
        print("For Loop no: ", length)
        for prb0, str0 in best_of_len[length - 1]:
            print('best_of_len: ', best_of_len)
            tokenized_text_cp = tokenized_text.copy()
            mask_ix_start = tokenized_text_cp.index('[MASK]')
            tokenized_text_cp[mask_ix_start:mask_ix_start + length - 1] = str0
            print("Text before processing ", tokenized_text_cp)
            res = evaluate_one_word(tokenized_text_cp, topk=eval_count)
            updated_res = [(update_probs(prb, prb0), str0 + [char]) for prb, char in res.items()]
            if length not in best_of_len:
                best_of_len[length] = []
            best_of_len[length] += updated_res
            best_of_len[length] = sorted(best_of_len[length], key=lambda x: x[0])
        best_of_len[length] = best_of_len[length][:beam_size]
    return best_of_len


if __name__ == "__main__":
    text = "[CLS] 把 台 上 几 个 原  本 羞 却 [MASK] 的 男 孩 们 炒 成 了 热 门 的 幕 间 演 出 乐 队 。 [SEP] 他 们 就 这 样 学 会 了 如 何 抓 住 持 续 增 长 的 听 众 。 [SEP]"
    # Tokenize and generate masks of length 1,2,3,4

    tokenized_text = tokenizer.tokenize(text)
    mask_ix = tokenized_text.index('[MASK]')
    tokenized_texts = []
    for i in range(4):
        tokenized_text_cp = tokenized_text.copy()
        tokenized_text_cp[mask_ix:mask_ix] = ['[MASK]'] * i
        tokenized_texts += [tokenized_text_cp]

    tokenized_text = tokenized_texts[3]

    result = stepwise_beam_search(tokenized_text)
    print(result)