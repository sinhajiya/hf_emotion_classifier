import torch
import os
import time
import evaluate
import numpy as np

def print_model_params(model):
    for name, param in model.named_parameters():
        print(f"{name:<60}  shape={tuple(param.shape)}  dtype={param.dtype}")


def get_model_size(model):
    """Calculates the model size in megabytes."""
    torch.save(model.state_dict(), "temp_model.p")
    size_mb = os.path.getsize("temp_model.p") / (1024 * 1024)
    os.remove("temp_model.p")
    return size_mb

def measure_latency(model, dataset, num_samples=100):
    model.eval()
    device = next(model.parameters()).device
    test_subset = dataset.select(range(num_samples))
    inputs = []
    for i in range(num_samples):
        single = {}
        for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            if key in test_subset[i]:
                single[key] = torch.tensor(test_subset[i][key]).unsqueeze(0).to(device)
        inputs.append(single)

    total_time = 0.0
    for i in range(num_samples):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs[i])
        end_time = time.time()
        total_time += (end_time - start_time)
    return (total_time / num_samples) * 1000

def acc_f1(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}
