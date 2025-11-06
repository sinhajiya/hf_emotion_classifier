from datasets import load_dataset
from transformers import AutoTokenizer


def emotions(mode=None):
    ds = load_dataset("dair-ai/emotion", "split")
    if mode != None:
        return (ds[mode])
    else:
        return ds
    

def labels_and_ids():
    id2labels = {0: 'sadness',
             1: 'joy',
             2: 'love',
             3: 'anger',
             4: 'fear',
             5: 'surprise'}

    labels2id = {
        'sadness': 0,
        'joy' : 1,
        'love': 2,
        'anger':3,
        'fear' : 4,
        'surprise' : 5
    }

    return {'i2l': id2labels, 'l2i': labels2id}



def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)