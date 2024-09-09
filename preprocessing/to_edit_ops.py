import difflib

SAVE = 'equal'
REMOVE = 'delete'
REPLACE = 'replace'
ADD = 'insert'
START_TEOKEN = '$'
OP_TO_ID = {SAVE: 0, REMOVE: 1, REPLACE: 2, ADD: 3}


def get_edit_operations(src, tgt,to_id=False):
    matcher = difflib.SequenceMatcher(None, src, tgt)
    opcodes = matcher.get_opcodes()
    operations = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == SAVE:
            operations.extend([SAVE] * (i2 - i1))
        elif tag == REMOVE:
            operations.extend([REMOVE] * (i2 - i1))
        elif tag == REPLACE:
            operations.append(REPLACE)
            if i2 - i1 > 1:
                operations.extend([REMOVE] * (i2 - i1 - 1))

        elif tag == ADD:
            if operations[-1] == SAVE:
                operations[-1] = ADD
            elif operations[-1] == REMOVE:
                operations[-1] = REPLACE
            else:
                raise ValueError("Invalid operation sequence")
    if to_id:
        operations = [OP_TO_ID[op] for op in operations]
    return operations


# if run as main:
if __name__ == "__main__":
    from tokenizer import encode_bos_eos_pad
    from transformers import PreTrainedTokenizerFast
    import Levenshtein

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../data/tokenizer.json", model_max_length=128)
    src = "../data/USPTO-MIT_RtoP_aug5/train/src-train.txt"
    tgt = "../data/USPTO-MIT_RtoP_aug5/train/tgt-train.txt"
    with open(src) as f:
        src_lines = f.read().splitlines()
    with open(tgt) as f:
        tgt_lines = f.read().splitlines()
    assert len(src_lines) == len(tgt_lines)
    for src, tgt in zip(src_lines, tgt_lines):
        src_tokens = [tokenizer.bos_token_id] + tokenizer.encode(src, add_special_tokens=False, truncation=False)
        # print(src)
        # print(src_tokens)
        tgt_tokens = [tokenizer.bos_token_id] + tokenizer.encode(tgt, add_special_tokens=False, truncation=False)
        # print(tgt)
        # print(tgt_tokens)
        ops = get_edit_operations(src_tokens, tgt_tokens)
        eq_ops = sum([x == SAVE for x in ops])

        ops = [o[0] for o in ops]
        print(Levenshtein.distance(src_tokens, tgt_tokens) / len(src_tokens), eq_ops / len(ops), ops)
