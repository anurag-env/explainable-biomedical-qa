"""
Staged Training Entrypoint.

Implements 5-stage curriculum training with Optuna hyperparameter optimization:
  Stage 1: PubMedQA pretraining (273,500 samples)
  Stage 2: MedQA fine-tuning (12,723 samples)
  Stage 3: HealthSearchQA fine-tuning (4,440 samples)
  Stage 4: Hindi translation fine-tuning
  Stage 5: Voice generation training

Usage:
  python -m src.train --stage 1
  python -m src.train --stage 2 --resume outputs/checkpoints/stage1/best
  python -m src.train --optuna --n-trials 50
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.models.rerb_qa import RERBQAModel
from src.data.dataset import BiomedicalQADataset, create_dataloaders
from src.data.preprocessing import BiomedicalPreprocessor


def load_config(config_dir="configs"):
    """Load all YAML configuration files."""
    configs = {}
    for name in ["data", "model", "train"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path) as f:
                configs[name] = yaml.safe_load(f)
    return configs


def get_optimizer(model, name, lr, weight_decay):
    """Create optimizer by name."""
    lr = float(lr)
    weight_decay = float(weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("Lion not available, falling back to AdamW")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "ranger":
        try:
            from ranger21 import Ranger21
            return Ranger21(model.parameters(), lr=lr, weight_decay=weight_decay, num_epochs=10, num_batches_per_epoch=100)
        except ImportError:
            print("Ranger not available, falling back to AdamW")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, name, num_training_steps, warmup_ratio=0.1):
    """Create learning rate scheduler using PyTorch native schedulers."""
    num_warmup = int(num_training_steps * warmup_ratio)

    # Warmup: linearly scale LR from 0 to base LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(num_warmup, 1)
    )

    # Cosine decay after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_training_steps - num_warmup, 1)
    )

    # Chain: warmup then cosine
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup],
    )


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    writer=None, grad_accum_steps=4, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, labels=labels)
        loss = output["loss"] / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += output["loss"].item()
        num_batches += 1

        if step % 100 == 0:
            print(f"  Epoch {epoch} | Step {step} | Loss: {output['loss'].item():.4f}")

        if writer:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss", output["loss"].item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attention_mask, labels=labels)

        if "loss" in output:
            total_loss += output["loss"].item()

        preds = output["logits"].argmax(dim=-1)
        valid = labels >= 0
        correct += (preds[valid] == labels[valid]).sum().item()
        total += valid.sum().item()

        all_preds.extend(preds[valid].cpu().tolist())
        all_labels.extend(labels[valid].cpu().tolist())

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(len(dataloader), 1)

    # Compute F1
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0,
    }
    return metrics


def train_stage(stage, configs, model=None, resume_from=None, device="cuda"):
    """Run a single training stage."""
    train_cfg = configs["train"]["stages"][stage]
    model_cfg = configs["model"]
    data_cfg = configs["data"]
    general_cfg = configs["train"].get("training", {})

    print(f"\n{'='*60}")
    print(f"Stage {stage}: {train_cfg['name']}")
    print(f"{'='*60}")

    # Determine dataset
    dataset_name = train_cfg.get("dataset")
    if dataset_name is None:
        print(f"Stage {stage} has no dataset — skipping QA training.")
        return model, {}

    ds_cfg = data_cfg["datasets"][dataset_name]

    # Create model if not provided
    if model is None:
        model = RERBQAModel(
            encoder_name=model_cfg["encoder"]["name"],
            hidden_size=model_cfg["encoder"]["hidden_size"],
            num_labels=model_cfg["classifier"]["num_labels"],
            num_heads=model_cfg["attention"]["num_heads"],
            max_iterations=model_cfg["recursive_refinement"]["max_iterations"],
            convergence_epsilon=model_cfg["recursive_refinement"]["convergence_epsilon"],
            dropout=model_cfg["classifier"]["dropout"],
        )

    # Resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        state_dict = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    # Create dataloaders
    data_path = ds_cfg["processed_dir"] if os.path.exists(ds_cfg["processed_dir"]) else ds_cfg["raw_dir"]
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=train_cfg["batch_size"],
        max_length=data_cfg["preprocessing"]["max_question_length"],
        tokenizer_name=data_cfg["preprocessing"]["tokenizer"],
        train_ratio=ds_cfg["splits"]["train"],
        val_ratio=ds_cfg["splits"]["val"],
        test_ratio=ds_cfg["splits"]["test"],
        seed=general_cfg.get("seed", 42),
    )

    # Setup training
    optimizer = get_optimizer(
        model, general_cfg.get("optimizer", "adamw"),
        train_cfg["learning_rate"], general_cfg.get("weight_decay", 0.01),
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = get_scheduler(
        optimizer, general_cfg.get("scheduler", "cosine"),
        total_steps, train_cfg.get("warmup_ratio", 0.1),
    )

    # Tensorboard
    log_dir = os.path.join(general_cfg.get("log_dir", "outputs/logs"), f"stage{stage}")
    writer = SummaryWriter(log_dir)

    # Checkpoint directory
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    patience = general_cfg.get("early_stopping_patience", 5)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            writer=writer,
            grad_accum_steps=general_cfg.get("gradient_accumulation_steps", 4),
            max_grad_norm=general_cfg.get("max_grad_norm", 1.0),
        )

        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch}/{train_cfg['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val MCC: {val_metrics['mcc']:.4f}")

        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1_macro"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)

        # Save best checkpoint
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  -> New best! Saved to {ckpt_dir / 'best.pt'}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    # Final evaluation on test set
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results (Stage {stage}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump({"val_best": best_accuracy, "test": test_metrics}, f, indent=2)

    writer.close()
    return model, test_metrics


def run_optuna_study(configs, device="cuda"):
    """Run Optuna hyperparameter optimization."""
    import optuna

    optuna_cfg = configs["train"]["optuna"]

    def objective(trial):
        # Sample hyperparameters from search space
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_int("epochs", 5, 15)  # Reduce for search
        num_heads = trial.suggest_categorical("attention_heads", [8, 12, 16])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
        max_iters = trial.suggest_int("recursive_iterations", 3, 10)
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw"])

        # Override configs
        configs["model"]["attention"]["num_heads"] = num_heads
        configs["model"]["recursive_refinement"]["max_iterations"] = max_iters
        configs["model"]["classifier"]["dropout"] = dropout
        configs["train"]["stages"][1]["learning_rate"] = lr
        configs["train"]["stages"][1]["batch_size"] = batch_size
        configs["train"]["stages"][1]["epochs"] = epochs
        configs["train"]["training"]["weight_decay"] = weight_decay
        configs["train"]["training"]["optimizer"] = optimizer_name

        try:
            _, metrics = train_stage(1, configs, device=device)
            return metrics.get("accuracy", 0.0)
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(
        study_name=optuna_cfg.get("study_name", "rerb_qa"),
        direction=optuna_cfg.get("direction", "maximize"),
        storage=optuna_cfg.get("storage"),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=optuna_cfg.get("n_trials", 50))

    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Save results
    results_path = Path("outputs/reports")
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "optuna_results.json", "w") as f:
        json.dump({
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "n_trials": len(study.trials),
        }, f, indent=2)

    return study


def main():
    parser = argparse.ArgumentParser(description="RAG-RERoBERTa Training")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Training stage (1-5)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Configuration directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")
    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of Optuna trials")
    parser.add_argument("--all-stages", action="store_true",
                        help="Run all 5 stages sequentially")
    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load configs
    configs = load_config(args.config_dir)

    # Set seed
    seed = configs["train"].get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.optuna:
        if args.n_trials:
            configs["train"]["optuna"]["n_trials"] = args.n_trials
        run_optuna_study(configs, device)
    elif args.all_stages:
        model = None
        for stage in range(1, 6):
            resume = None
            if stage > 1:
                prev_ckpt = configs["train"]["stages"][stage].get("resume_from")
                if prev_ckpt and os.path.exists(f"{prev_ckpt}.pt"):
                    resume = f"{prev_ckpt}.pt"
            model, _ = train_stage(stage, configs, model=model,
                                   resume_from=resume, device=device)
    elif args.stage:
        train_stage(args.stage, configs, resume_from=args.resume, device=device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
