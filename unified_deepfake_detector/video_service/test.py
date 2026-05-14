import os
import json
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from models.fusion import SpatialTemporalFusion
from utils.dataset import DeepfakeDataset
from utils.inference import classify_probability
from config import (
    SEQUENCE_LENGTH,
    DATASET_PREPROCESSED,
    LOG_DIR,
    NUM_WORKERS,
    USE_AUDIO_BRANCH,
    INFERENCE_UNCERTAINTY_MARGIN,
)


FACEFORENSICS_FAKE_PREFIXES = {
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
}


def infer_dataset_name(sample_name):
    if sample_name.isdigit():
        return "FaceForensics++"
    if sample_name.startswith("wild"):
        return "WildDeepfake"
    if re.match(r"^id\d+", sample_name):
        return "Celeb-DF v2"

    prefix = sample_name.split("_", 1)[0]
    if prefix in FACEFORENSICS_FAKE_PREFIXES:
        return "FaceForensics++"

    return "Unknown"


def compute_binary_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_acc = (real_recall + fake_recall) / 2.0
    macro_f1 = (
        f1_score(labels, preds, pos_label=0, average="binary", zero_division=0)
        + f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    ) / 2.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    data_dir,
    model_path,
    batch_size=4,
    output_dir=None,
    num_workers=NUM_WORKERS,
    use_audio=USE_AUDIO_BRANCH,
    uncertainty_margin=INFERENCE_UNCERTAINTY_MARGIN,
):
    if output_dir is None:
        output_dir = LOG_DIR
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    model = SpatialTemporalFusion(seq_length=SEQUENCE_LENGTH, use_audio=use_audio).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = DeepfakeDataset(
        os.path.join(data_dir, "test"),
        sequence_length=SEQUENCE_LENGTH,
        phase="val",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_preds = []
    all_labels = []
    all_probs = []
    sample_names = [os.path.basename(path) for path in test_dataset.video_paths]

    print("Running evaluation...")
    with torch.no_grad():
        for frames, labels, masks, audio_features in tqdm(test_loader):
            frames = frames.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            audio_features = audio_features.to(device)

            with torch.amp.autocast(str(device)):
                logits = model(frames, mask=masks, audio_features=audio_features)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    thresholds_to_try = np.arange(0.1, 0.95, 0.01)
    default_metrics = compute_binary_metrics(all_labels, all_preds, all_probs)
    best_thresh = 0.5
    best_preds = all_preds
    best_f1 = default_metrics["f1_score"]
    threshold_rows = []

    for threshold in thresholds_to_try:
        preds_t = [1 if p >= threshold else 0 for p in all_probs]
        threshold_metrics = compute_binary_metrics(all_labels, preds_t, all_probs)
        threshold_rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": round(threshold_metrics["accuracy"], 4),
            "precision": round(threshold_metrics["precision"], 4),
            "recall": round(threshold_metrics["recall"], 4),
            "f1_score": round(threshold_metrics["f1_score"], 4),
            "real_recall": round(threshold_metrics["real_recall"], 4),
            "fake_recall": round(threshold_metrics["fake_recall"], 4),
            "balanced_accuracy": round(threshold_metrics["balanced_accuracy"], 4),
            "macro_f1": round(threshold_metrics["macro_f1"], 4),
        })

        if threshold_metrics["f1_score"] > best_f1:
            best_f1 = threshold_metrics["f1_score"]
            best_thresh = round(float(threshold), 2)
            best_preds = preds_t

    best_balanced_row = max(threshold_rows, key=lambda row: (row["macro_f1"], row["balanced_accuracy"]))
    optimal_metrics = compute_binary_metrics(all_labels, best_preds, all_probs)

    class_report = classification_report(
        all_labels,
        best_preds,
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0,
    )

    dataset_rows = []
    error_cases = []
    prediction_audit = []
    for sample_name, label, prob, pred in zip(sample_names, all_labels, all_probs, best_preds):
        dataset_name = infer_dataset_name(sample_name)
        classification = classify_probability(prob, threshold=best_thresh, uncertainty_margin=uncertainty_margin)
        dataset_rows.append({
            "sample_name": sample_name,
            "dataset": dataset_name,
            "label": int(label),
            "prob_fake": float(prob),
            "pred": int(pred),
        })
        prediction_audit.append({
            "sample_name": sample_name,
            "dataset": dataset_name,
            "label": "FAKE" if int(label) == 1 else "REAL",
            "pred": "FAKE" if int(pred) == 1 else "REAL",
            "prob_fake": round(float(prob), 4),
            "distance_from_threshold": classification["distance_from_threshold"],
            "decision_band": classification["decision_band"],
            "uncertain": classification["prediction"] == "UNCERTAIN",
        })

        if int(pred) != int(label):
            error_cases.append({
                "sample_name": sample_name,
                "dataset": dataset_name,
                "label": "FAKE" if int(label) == 1 else "REAL",
                "pred": "FAKE" if int(pred) == 1 else "REAL",
                "prob_fake": round(float(prob), 4),
                "error_type": "false_positive" if int(label) == 0 else "false_negative",
                "confidence_gap": round(abs(float(prob) - best_thresh), 4),
                "decision_band": classification["decision_band"],
            })

    dataset_breakdown = {}
    for dataset_name in sorted({row["dataset"] for row in dataset_rows}):
        subset = [row for row in dataset_rows if row["dataset"] == dataset_name]
        subset_labels = [row["label"] for row in subset]
        subset_probs = [row["prob_fake"] for row in subset]
        subset_preds = [row["pred"] for row in subset]
        subset_metrics = compute_binary_metrics(subset_labels, subset_preds, subset_probs)

        dataset_breakdown[dataset_name] = {
            "samples": len(subset),
            "real_samples": int(sum(1 for x in subset_labels if x == 0)),
            "fake_samples": int(sum(1 for x in subset_labels if x == 1)),
            "accuracy": round(subset_metrics["accuracy"], 4),
            "precision": round(subset_metrics["precision"], 4),
            "recall": round(subset_metrics["recall"], 4),
            "f1_score": round(subset_metrics["f1_score"], 4),
            "roc_auc": round(subset_metrics["roc_auc"], 4),
            "real_recall": round(subset_metrics["real_recall"], 4),
            "fake_recall": round(subset_metrics["fake_recall"], 4),
            "balanced_accuracy": round(subset_metrics["balanced_accuracy"], 4),
            "macro_f1": round(subset_metrics["macro_f1"], 4),
        }

    error_cases.sort(key=lambda row: row["confidence_gap"], reverse=True)
    false_positives = [row for row in error_cases if row["error_type"] == "false_positive"]
    false_negatives = [row for row in error_cases if row["error_type"] == "false_negative"]
    uncertain_cases = [row for row in prediction_audit if row["uncertain"]]

    score_bands = {
        "very_low_fake_probability": sum(1 for p in all_probs if p < 0.1),
        "low_fake_probability": sum(1 for p in all_probs if 0.1 <= p < 0.25),
        "borderline": sum(1 for p in all_probs if best_thresh - uncertainty_margin <= p <= best_thresh + uncertainty_margin),
        "moderate_fake_probability": sum(1 for p in all_probs if 0.25 <= p < 0.6),
        "high_fake_probability": sum(1 for p in all_probs if p >= 0.6),
    }

    metrics = {
        "accuracy": round(optimal_metrics["accuracy"], 4),
        "precision": round(optimal_metrics["precision"], 4),
        "recall": round(optimal_metrics["recall"], 4),
        "f1_score": round(optimal_metrics["f1_score"], 4),
        "roc_auc": round(auc, 4),
        "optimal_threshold": best_thresh,
        "best_balanced_threshold": best_balanced_row["threshold"],
        "best_balanced_macro_f1": best_balanced_row["macro_f1"],
        "best_balanced_real_recall": best_balanced_row["real_recall"],
        "best_balanced_fake_recall": best_balanced_row["fake_recall"],
        "real_recall_at_optimal_threshold": round(optimal_metrics["real_recall"], 4),
        "fake_recall_at_optimal_threshold": round(optimal_metrics["fake_recall"], 4),
        "balanced_accuracy_at_optimal_threshold": round(optimal_metrics["balanced_accuracy"], 4),
        "macro_f1_at_optimal_threshold": round(optimal_metrics["macro_f1"], 4),
        "uncertainty_margin": uncertainty_margin,
        "uncertain_case_count": len(uncertain_cases),
        "total_samples": len(all_labels),
        "total_real": int(all_labels.count(0.0)),
        "total_fake": int(all_labels.count(1.0)),
        "score_bands": score_bands,
        "per_class": class_report,
        "dataset_breakdown": dataset_breakdown,
        "model_path": model_path,
        "device": str(device),
    }

    print("\n========= EVALUATION RESULTS =========")
    print(f"  Accuracy:           {optimal_metrics['accuracy']:.4f}")
    print(f"  Precision:          {optimal_metrics['precision']:.4f}")
    print(f"  Recall:             {optimal_metrics['recall']:.4f}")
    print(f"  F1 Score:           {optimal_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:            {auc:.4f}")
    print(f"  Optimal Threshold:  {best_thresh}")
    print(f"  Real Recall:        {optimal_metrics['real_recall']:.4f}")
    print(f"  Fake Recall:        {optimal_metrics['fake_recall']:.4f}")
    print(f"  Balanced Threshold: {best_balanced_row['threshold']}")

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    threshold_path = os.path.join(output_dir, "threshold_sweep.json")
    with open(threshold_path, "w") as f:
        json.dump(threshold_rows, f, indent=2)

    dataset_path = os.path.join(output_dir, "dataset_breakdown.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset_breakdown, f, indent=2)

    errors_path = os.path.join(output_dir, "error_cases.json")
    with open(errors_path, "w") as f:
        json.dump({
            "false_positives": false_positives[:100],
            "false_negatives": false_negatives[:100],
            "uncertain_cases": uncertain_cases[:100],
        }, f, indent=2)

    audit_path = os.path.join(output_dir, "prediction_audit.json")
    with open(audit_path, "w") as f:
        json.dump(prediction_audit, f, indent=2)

    score_bands_path = os.path.join(output_dir, "score_bands.json")
    with open(score_bands_path, "w") as f:
        json.dump(score_bands, f, indent=2)

    cm = confusion_matrix(all_labels, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    if len(set(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC={auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

        prec_vals, rec_vals, _ = precision_recall_curve(all_labels, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_vals, prec_vals, color="#ff7f0e", linewidth=2, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        pr_path = os.path.join(output_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()

        print(f"Plots saved: {cm_path}, {roc_path}, {pr_path}")
    else:
        print(f"Plots saved: {cm_path} (ROC/PR skipped - single class in test set)")

    print(f"Extra reports saved: {threshold_path}, {dataset_path}, {errors_path}, {audit_path}, {score_bands_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATASET_PREPROCESSED)
    parser.add_argument("--weights", type=str, default="logs/best_model.pth")
    parser.add_argument("--output_dir", type=str, default=LOG_DIR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--use_audio", action="store_true", default=USE_AUDIO_BRANCH)
    parser.add_argument("--uncertainty_margin", type=float, default=INFERENCE_UNCERTAINTY_MARGIN)
    args = parser.parse_args()

    evaluate_model(
        args.data_dir,
        args.weights,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_audio=args.use_audio,
        uncertainty_margin=args.uncertainty_margin,
    )
