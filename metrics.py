import os
import time
import torch
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def print_model_params(model):
    """Prints all model parameters, shapes, and dtypes."""
    for name, param in model.named_parameters():
        print(f"{name:<60}  shape={tuple(param.shape)}  dtype={param.dtype}")



def get_model_size(model):
    """Calculates model size in megabytes (MB)."""
    tmp_path = "temp_model.p"
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
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



def compute_classification_metrics(predictions, labels, num_classes=None):
    """
    Computes classification metrics:
      - accuracy
      - macro F1
      - per-class F1 (list)
      - confusion matrix
    """

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    macro_f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    per_class_f1_dict = metric_f1.compute(predictions=predictions, references=labels, average=None)
    per_class_f1 = per_class_f1_dict["f1"]

    cm = confusion_matrix(labels, predictions, labels=range(num_classes) if num_classes else None)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def acc_f1(eval_pred):
    """
    For Hugging Face Trainer compute_metrics argument.
    
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}




def summarize_evaluation(
    model,
    dataset,
    predictions,
    labels,
    model_name="Model",
    num_classes=6,
    class_names=["sadness", "joy", "love", "anger", "fear", "surprise"]
):
    """
      Accuracy
      Macro F1
      Per-class F1
      Classification report (precision, recall, F1)
      Confusion matrix 
      Model size (MB)
      Inference latency (ms/sample)
    """
    print(f"\n{model_name} Evaluation Summary\n{'='*60}")

    metrics = compute_classification_metrics(predictions, labels, num_classes)
    model_size = get_model_size(model)
    latency = measure_latency(model, dataset)

    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Model Size:     {model_size:.2f} MB")
    print(f"Latency:        {latency:.2f} ms/sample\n")

    print(" Classification Report:")
    print(classification_report(labels, predictions, target_names=class_names, digits=3))

    print("Per-Class F1 Scores:")
    for i, f1_score in enumerate(metrics["per_class_f1"]):
        label_name = class_names[i] if class_names else f"Class {i}"
        print(f"  {label_name:<15}: {f1_score:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=class_names if class_names else np.arange(num_classes),
        yticklabels=class_names if class_names else np.arange(num_classes)
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "model_size_mb": model_size,
        "latency_ms": latency,
    }
