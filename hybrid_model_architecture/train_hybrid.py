import numpy as np
import argparse
import os
from datetime import datetime
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json



# Import our modules
from hybrid_model import (
    HybridMultiTaskModel,
    compute_multitask_loss,
    validate_span_indices,
    check_span_availability_consistency
)
from hybrid_dataset import SpanAwareDataset, collate_fn

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# TRAINING & EVALUATION
def train_epoch(model, dataloader, optimizer, device, classification_weight, span_weight, entropy_weight):
    model.train()

    total_loss = 0
    total_classification_loss = 0
    total_span_loss = 0
    total_entropy_loss = 0
    total_spans_used = 0

    for batch in tqdm(dataloader, desc="Training", mininterval=5, leave=False):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        token_span_start = batch['token_span_start'].to(device)
        token_span_end = batch['token_span_end'].to(device)
        span_available = batch['span_available'].to(device)
        entropy = batch['entropy'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        losses = compute_multitask_loss(
            outputs,
            labels,
            token_span_start,
            token_span_end,
            span_available,
            entropy,
            classification_weight=classification_weight,
            span_weight=span_weight,
            entropy_weight=entropy_weight
        )

        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += losses['total_loss'].item()
        total_classification_loss += losses['classification_loss'].item()
        total_span_loss += losses['span_loss'].item()
        total_entropy_loss += losses['entropy_loss'].item()
        total_spans_used += losses['num_spans_used']

    num_batches = len(dataloader)

    return {
        'total_loss': total_loss / num_batches,
        'classification_loss': total_classification_loss / num_batches,
        'span_loss': total_span_loss / num_batches,
        'entropy_loss': total_entropy_loss / num_batches,
        'spans_used': total_spans_used
    }


def evaluate(model, dataloader, device, classification_weight, span_weight, entropy_weight):
    model.eval()

    total_loss = 0
    total_classification_loss = 0
    total_span_loss = 0
    total_entropy_loss = 0
    total_spans_used = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_span_start = batch['token_span_start'].to(device)
            token_span_end = batch['token_span_end'].to(device)
            span_available = batch['span_available'].to(device)
            entropy = batch['entropy'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Compute loss
            losses = compute_multitask_loss(
                outputs,
                labels,
                token_span_start,
                token_span_end,
                span_available,
                entropy,
                classification_weight=classification_weight,
                span_weight=span_weight,
                entropy_weight=entropy_weight
            )

            # Predictions (classification only)
            preds = torch.argmax(outputs['classification_logits'], dim=1)

            # Accumulate
            total_loss += losses['total_loss'].item()
            total_classification_loss += losses['classification_loss'].item()
            total_span_loss += losses['span_loss'].item()
            total_entropy_loss += losses['entropy_loss'].item()
            total_spans_used += losses['num_spans_used']

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    num_batches = len(dataloader)

    return {
        'total_loss': total_loss / num_batches,
        'classification_loss': total_classification_loss / num_batches,
        'span_loss': total_span_loss / num_batches,
        'entropy_loss': total_entropy_loss / num_batches,
        'spans_used': total_spans_used,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# TRAINER CLASS
class HybridTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize tokenizer
        print("INITIALIZING TOKENIZER")
        self.tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')
        print("CodeBERT tokenizer (Fast) loaded")

        # Load datasets
        print("LOADING DATASETS")

        print("\nTrain Dataset:")
        train_dataset = SpanAwareDataset(args.train, self.tokenizer, args.max_length)

        print("\nValidation Dataset:")
        val_dataset = SpanAwareDataset(args.val, self.tokenizer, args.max_length)

        print("\nTest Dataset:")
        test_dataset = SpanAwareDataset(args.test, self.tokenizer, args.max_length)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Validate first batch
        print("VALIDATING DATA")
        self._validate_first_batch()

        # Initialize model
        print("INITIALIZING MODEL")
        self.model = HybridMultiTaskModel(dropout=args.dropout).to(self.device)

        trainable, total = self.model.get_trainable_parameters()
        print(f"Model: HybridMultiTaskModel")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Device: {self.device}")

        # History
        self.history = {
            'train_loss': [],
            'train_classification_loss': [],
            'train_span_loss': [],
            'train_entropy_loss': [],
            'val_loss': [],
            'val_classification_loss': [],
            'val_span_loss': [],
            'val_entropy_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        self._try_resume()


    def _validate_first_batch(self):
        batch = next(iter(self.train_loader))

        print("\nValidating first batch:")
        print(f"  Batch size: {len(batch['label'])}")

        # Check span indices
        is_valid = validate_span_indices(
            batch['token_span_start'],
            batch['token_span_end'],
            batch['span_available'],
            self.args.max_length
        )

        if not is_valid:
            raise ValueError("Span validation failed!")

        print("  Span indices valid")

        # Check consistency
        is_consistent = check_span_availability_consistency(
            batch['label'],
            batch['span_available']
        )

        if not is_consistent:
            print("  Consistency warning")
        else:
            print("  Span availability consistent")

    def train(self):
        print("TRAINING HYBRID MULTI-TASK MODEL")

        # Stage 1: Warm-up
        if self.args.warmup_epochs > 0:
            print("\n STAGE 1: WARM-UP (Task heads only)")

            self.model.freeze_encoder()
            trainable, _ = self.model.get_trainable_parameters()
            print(f"Trainable parameters: {trainable:,}")

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.warmup_lr
            )

            start = self.start_epoch if self.resume_stage == "warmup" else 0
            for epoch in range(self.args.warmup_epochs):
                self.save_checkpoint("last_checkpoint.pt", extra={
                    "epoch": epoch + 1,
                    "stage" : "warmup"
                })
                print(f"\nWarm-up Epoch {epoch + 1}/{self.args.warmup_epochs}")

                # Train
                train_metrics = train_epoch(
                    self.model, self.train_loader, optimizer, self.device,
                    self.args.warmup_classification_weight, self.args.warmup_span_weight, self.args.warmup_entropy_weight
                )

                # Validate
                val_metrics = evaluate(
                    self.model, self.val_loader, self.device,
                    self.args.warmup_classification_weight, self.args.warmup_span_weight, self.args.warmup_entropy_weight
                )

                # Log
                self._log_metrics(train_metrics, val_metrics, "warm-up")

        # Stage 2: Fine-tuning
        print("\n STAGE 2: FINE-TUNING (All layers)")

        self.model.unfreeze_encoder()
        trainable, _ = self.model.get_trainable_parameters()
        print(f"Trainable parameters: {trainable:,}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.finetune_lr
        )

        best_f1 = 0.0

        start = self.start_epoch if self.resume_stage == "finetune" else 0
        for epoch in range(self.args.finetune_epochs):
            self.save_checkpoint("last_checkpoint.pt", extra={
                "epoch": epoch + 1,
                "stage": "finetune"
            })

            print(f"\nFine-tune Epoch {epoch + 1}/{self.args.finetune_epochs}")

            # Train
            train_metrics = train_epoch(
                self.model, self.train_loader, optimizer, self.device,
                self.args.finetune_classification_weight, self.args.finetune_span_weight, self.args.finetune_entropy_weight
            )

            # Validate
            val_metrics = evaluate(
                self.model, self.val_loader, self.device,
                self.args.finetune_classification_weight, self.args.finetune_span_weight, self.args.finetune_entropy_weight
            )

            # Log
            self._log_metrics(train_metrics, val_metrics, "fine-tune")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint('best_model.pt')
                print(f"  Saved best model (F1: {best_f1:.4f})")

        print(f"Warm-up loss weights → "
              f"cls={self.args.warmup_classification_weight}, "
              f"span={self.args.warmup_span_weight}, "
              f"entropy={self.args.warmup_entropy_weight}")

        print(f"Fine-tune loss weights → "
              f"cls={self.args.finetune_classification_weight}, "
              f"span={self.args.finetune_span_weight}, "
              f"entropy={self.args.finetune_entropy_weight}")

        print("\nTraining complete!")

    def _log_metrics(self, train_metrics, val_metrics, stage):
        # Print
        print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
              f"(cls: {train_metrics['classification_loss']:.4f}, "
              f"span: {train_metrics['span_loss']:.4f}, "
              f"entropy: {train_metrics['entropy_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f} "
              f"(cls: {val_metrics['classification_loss']:.4f}, "
              f"span: {val_metrics['span_loss']:.4f}, "
              f"entropy: {val_metrics['entropy_loss']:.4f})")
        print(f"  Val Metrics: Acc={val_metrics['accuracy']:.4f}, "
              f"P={val_metrics['precision']:.4f}, "
              f"R={val_metrics['recall']:.4f}, "
              f"F1={val_metrics['f1']:.4f}")
        print(f"  Spans used (train/val): {train_metrics['spans_used']} / {val_metrics['spans_used']}")

        # Store
        self.history['train_loss'].append(train_metrics['total_loss'])
        self.history['train_classification_loss'].append(train_metrics['classification_loss'])
        self.history['train_span_loss'].append(train_metrics['span_loss'])
        self.history['train_entropy_loss'].append(train_metrics['entropy_loss'])
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['val_classification_loss'].append(val_metrics['classification_loss'])
        self.history['val_span_loss'].append(val_metrics['span_loss'])
        self.history['val_entropy_loss'].append(val_metrics['entropy_loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1'])

    def test(self):
        print("FINAL EVALUATION ON TEST SET")

        # Load best model
        best_model_path = os.path.join(self.args.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        test_metrics = evaluate(
            self.model, self.test_loader, self.device,
            self.args.finetune_classification_weight, self.args.finetune_span_weight, self.args.finetune_entropy_weight
        )

        # Print results
        print("\nTest Results:")
        print(f"Total Loss:           {test_metrics['total_loss']:.4f}")
        print(f"Classification Loss:  {test_metrics['classification_loss']:.4f}")
        print(f"Span Loss:            {test_metrics['span_loss']:.4f}")
        print(f"Entropy Loss:         {test_metrics['entropy_loss']:.4f}")
        print(f"Accuracy:             {test_metrics['accuracy']:.4f}")
        print(f"Precision:            {test_metrics['precision']:.4f}")
        print(f"Recall:               {test_metrics['recall']:.4f}")
        print(f"F1 Score:             {test_metrics['f1']:.4f}")
        print(f"Spans used:           {test_metrics['spans_used']}")

        # Save results
        results_path = os.path.join(self.args.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'test_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                 for k, v in test_metrics.items()},
                'history': self.history,
                'args': vars(self.args)
            }, f, indent=2)

        print(f"\nResults saved to {results_path}")

        return test_metrics

    def save_checkpoint(self, filename, epoch=None, stage=None):
        checkpoint_path = os.path.join(self.args.output_dir, filename)

        state = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            'epoch': epoch,
            'stage': stage
        }


        torch.save(state, checkpoint_path)


    def run(self):
        start_time = datetime.now()

        # Train
        self.train()

        # Test
        test_metrics = self.test()

        # Save final model
        self.save_checkpoint('final_model.pt')

        end_time = datetime.now()
        duration = end_time - start_time

        print("HYBRID MODEL TRAINING COMPLETE!")
        print(f"Total time: {duration}")
        print(f"Final Test F1: {test_metrics['f1']:.4f}")
        print(f"Models saved in: {self.args.output_dir}")

    def _try_resume(self):
        self.start_epoch = 0
        self.resume_stage = None

        ckpt_path = os.path.join(self.args.output_dir, "last_checkpoint.pt")

        if os.path.exists(ckpt_path):
            print("Resuming from checkpoint")
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.history = checkpoint.get("history", self.history)
            self.start_epoch = checkpoint.get("epoch", 0)
            self.resume_stage = checkpoint.get("stage")

            print(f"   Resuming from epoch {self.start_epoch}, stage={self.resume_stage}")
        else:
            print("No checkpoint found — starting fresh")



def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Multi-Task Model for API Key Detection"
    )

    # Data arguments
    parser.add_argument('--train', type=str, required=True,
                        help='Training CSV path')
    parser.add_argument('--val', type=str, required=True,
                        help='Validation CSV path')
    parser.add_argument('--test', type=str, required=True,
                        help='Test CSV path')

    # Model arguments
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Warm-up epochs (default: 2)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='Fine-tuning epochs (default: 5)')
    parser.add_argument('--warmup-lr', type=float, default=1e-3,
                        help='Warm-up learning rate (default: 1e-3)')
    parser.add_argument('--finetune-lr', type=float, default=2e-5,
                        help='Fine-tuning learning rate (default: 2e-5)')

    # Loss weights — Warm-up
    parser.add_argument('--warmup-classification-weight', type=float, default=1.0,
                        help='Warm-up classification loss weight')
    parser.add_argument('--warmup-span-weight', type=float, default=0.3,
                        help='Warm-up span loss weight')
    parser.add_argument('--warmup-entropy-weight', type=float, default=0.1,
                        help='Warm-up entropy loss weight')

    # Loss weights — Fine-tuning
    parser.add_argument('--finetune-classification-weight', type=float, default=1.0,
                        help='Finetune classification loss weight')
    parser.add_argument('--finetune-span-weight', type=float, default=0.5,
                        help='Finetune span loss weight')
    parser.add_argument('--finetune-entropy-weight', type=float, default=0.2,
                        help='Finetune entropy loss weight')

    # Output
    parser.add_argument('--output-dir', type=str, default='hybrid_outputs',
                        help='Output directory (default: hybrid_outputs)')

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Run training
    trainer = HybridTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()


    """
    
    RUN 1
    
    ================================================================================
HYBRID MULTI-TASK MODEL TRAINING
================================================================================

Configuration:
  train: data/train.csv
  val: data/val.csv
  test: data/test.csv
  max_length: 512
  dropout: 0.1
  batch_size: 16
  warmup_epochs: 2
  finetune_epochs: 5
  warmup_lr: 0.001
  finetune_lr: 2e-05
  classification_weight: 1.0
  span_weight: 1.0
  entropy_weight: 0.5
  output_dir: hybrid_outputs

================================================================================
INITIALIZING TOKENIZER
================================================================================
✓ CodeBERT tokenizer (Fast) loaded

================================================================================
LOADING DATASETS
================================================================================

Train Dataset:
  Loaded 5947 samples from data/train.csv
    Positive: 3306
    Negative: 2641
    With span annotations: 2075

Validation Dataset:
  Loaded 1274 samples from data/val.csv
    Positive: 708
    Negative: 566
    With span annotations: 444

Test Dataset:
  Loaded 1275 samples from data/test.csv
    Positive: 709
    Negative: 566
    With span annotations: 434

================================================================================
VALIDATING DATA
================================================================================

Validating first batch:
  Batch size: 16
  ✓ Span indices valid
  Positive samples with span: 5
  Positive samples without span: 5
  Negative samples with span: 0
  Negative samples without span: 6
ℹ️  Info: 5 positive samples without spans (50.0% - acceptable)
  ✓ Span availability consistent

================================================================================
INITIALIZING MODEL
================================================================================
Model: HybridMultiTaskModel
Total parameters: 124,943,109
Trainable parameters: 124,943,109
Device: cpu

================================================================================
TRAINING HYBRID MULTI-TASK MODEL
================================================================================

📌 STAGE 1: WARM-UP (Task heads only)
--------------------------------------------------------------------------------
✓ CodeBERT encoder frozen
Trainable parameters: 297,477

Warm-up Epoch 1/2
Training: 100%|███████████████████████████████████████████████████████████████████████████████| 372/372 [36:27<00:00,  5.88s/it]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████| 80/80 [03:07<00:00,  2.34s/it]
  Train Loss: nan (cls: 0.6273, span: nan, entropy: 3.2734)
  Val Loss: nan (cls: 0.5425, span: nan, entropy: 2.7183)
  Val Metrics: Acc=0.7559, P=0.8003, R=0.7472, F1=0.7728
  Spans used (train/val): 2075 / 444
  
  RUN 2
  
  ================================================================================
HYBRID MULTI-TASK MODEL TRAINING
================================================================================

Configuration:
  train: data/train.csv
  val: data/val.csv
  test: data/test.csv
  max_length: 512
  dropout: 0.1
  batch_size: 16
  warmup_epochs: 2
  finetune_epochs: 5
  warmup_lr: 0.001
  finetune_lr: 2e-05
  classification_weight: 1.0
  span_weight: 1.0
  entropy_weight: 0.5
  output_dir: hybrid_outputs

================================================================================
INITIALIZING TOKENIZER
================================================================================
✓ CodeBERT tokenizer (Fast) loaded

================================================================================
LOADING DATASETS
================================================================================

Train Dataset:
  Loaded 5947 samples from data/train.csv
    Positive: 3306
    Negative: 2641
    With span annotations: 2075

Validation Dataset:
  Loaded 1274 samples from data/val.csv
    Positive: 708
    Negative: 566
    With span annotations: 444

Test Dataset:
  Loaded 1275 samples from data/test.csv
    Positive: 709
    Negative: 566
    With span annotations: 434

================================================================================
VALIDATING DATA
================================================================================

Validating first batch:
  Batch size: 16
  ✓ Span indices valid
  Positive samples with span: 3
  Positive samples without span: 8
  Negative samples with span: 0
  Negative samples without span: 5
ℹ️  Info: 8 positive samples without spans (72.7% - acceptable)
  ✓ Span availability consistent

================================================================================
INITIALIZING MODEL
================================================================================
Model: HybridMultiTaskModel
Total parameters: 124,943,109
Trainable parameters: 124,943,109
Device: cpu

================================================================================
TRAINING HYBRID MULTI-TASK MODEL
================================================================================

📌 STAGE 1: WARM-UP (Task heads only)
--------------------------------------------------------------------------------
✓ CodeBERT encoder frozen
Trainable parameters: 297,477

Warm-up Epoch 1/2
Training: 100%|███████████████████████████████| 372/372 [35:39<00:00,  5.75s/it]
Evaluating: 100%|███████████████████████████████| 80/80 [03:07<00:00,  2.34s/it]
  Train Loss: 4.6471 (cls: 0.6184, span: 2.3914, entropy: 3.2744)
  Val Loss: 2.6742 (cls: 0.5282, span: 0.9118, entropy: 2.4686)
  Val Metrics: Acc=0.8061, P=0.7794, R=0.9082, F1=0.8389
  Spans used (train/val): 2075 / 444

  """