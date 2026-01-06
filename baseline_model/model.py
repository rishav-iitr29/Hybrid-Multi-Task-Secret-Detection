import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# Setting environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# DATASET
class BaselineDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"  Loaded {len(self.df)} samples from {csv_path}")
        print(f"  Positive samples: {(self.df['has_secret'] == 1).sum()}")
        print(f"  Negative samples: {(self.df['has_secret'] == 0).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        code_snippet = str(row['code_snippet'])
        has_secret = int(row['has_secret'])

        # Tokenize
        encoding = self.tokenizer(
            code_snippet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0).long(),
            'attention_mask': encoding['attention_mask'].squeeze(0).long(),
            'label': torch.tensor(has_secret, dtype=torch.long)
        }


# MODEL
class BaselineCodeBERTClassifier(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base', dropout=0.1):
        super(BaselineCodeBERTClassifier, self).__init__()

        self.codebert = RobertaModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary: 0 or 1
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        logits = self.classifier(pooled_output)
        return logits

    def freeze_encoder(self):
        # Freeze CodeBERT for warm-up
        for param in self.codebert.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        # Unfreeze CodeBERT for fine-tuning
        for param in self.codebert.parameters():
            param.requires_grad = True


# TRAINING & EVALUATION
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Predictions
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


# MAIN TRAINING PIPELINE
class BaselineTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize tokenizer
        print("INITIALIZING TOKENIZER")
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        print("CodeBERT tokenizer loaded")

        # Load datasets
        print("LOADING DATASETS")

        print("\nTrain Dataset:")
        train_dataset = BaselineDataset(args.train, self.tokenizer, args.max_length)

        print("\nValidation Dataset:")
        val_dataset = BaselineDataset(args.val, self.tokenizer, args.max_length)

        print("\nTest Dataset:")
        test_dataset = BaselineDataset(args.test, self.tokenizer, args.max_length)

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Initialize model
        print("INITIALIZING MODEL")
        self.model = BaselineCodeBERTClassifier(dropout=args.dropout).to(self.device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Model: BaselineCodeBERTClassifier")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Device: {self.device}")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

    def train(self):
        print("TRAINING BASELINE CLASSIFIER")

        # Stage 1: Warm-up (freeze encoder, train classifier only)
        if self.args.warmup_epochs > 0:
            print("\nSTAGE 1: WARM-UP (Classifier only)")

            self.model.freeze_encoder()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable:,}")

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.warmup_lr
            )

            for epoch in range(self.args.warmup_epochs):
                print(f"\nWarm-up Epoch {epoch + 1}/{self.args.warmup_epochs}")

                # Train
                train_loss = train_epoch(
                    self.model, self.train_loader, optimizer,
                    self.criterion, self.device
                )

                # Validate
                val_metrics = evaluate(
                    self.model, self.val_loader,
                    self.criterion, self.device
                )

                # Log
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
                self.history['val_f1'].append(val_metrics['f1'])

                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")

        # Stage 2: Fine-tuning (unfreeze all)
        print("\nSTAGE 2: FINE-TUNING (All layers)")

        self.model.unfreeze_encoder()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.finetune_lr
        )

        best_f1 = 0.0

        for epoch in range(self.args.finetune_epochs):
            print(f"\nFine-tune Epoch {epoch + 1}/{self.args.finetune_epochs}")

            # Train
            train_loss = train_epoch(
                self.model, self.train_loader, optimizer,
                self.criterion, self.device
            )

            # Validate
            val_metrics = evaluate(
                self.model, self.val_loader,
                self.criterion, self.device
            )

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_checkpoint('best_model.pt')
                print(f"  Saved best model (F1: {best_f1:.4f})")

        print("\nTraining complete!")

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
            self.model, self.test_loader,
            self.criterion, self.device
        )

        # Print results
        print("\nTest Results:")
        print(f"Loss:      {test_metrics['loss']:.4f}")
        print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"F1 Score:  {test_metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        cm = test_metrics['confusion_matrix']
        print(f"  True Negatives:  {cm['tn']}")
        print(f"  False Positives: {cm['fp']}")
        print(f"  False Negatives: {cm['fn']}")
        print(f"  True Positives:  {cm['tp']}")

        # Save results
        results_path = os.path.join(self.args.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'test_metrics': test_metrics,
                'history': self.history
            }, f, indent=2)

        print(f"\nResults saved to {results_path}")

        return test_metrics

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.args.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, checkpoint_path)

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

        print("BASELINE TRAINING COMPLETE!")
        print(f"Total time: {duration}")
        print(f"Final Test F1: {test_metrics['f1']:.4f}")
        print(f"Models saved in: {self.args.output_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline CodeBERT Classifier for API Key Detection"
    )

    # Data arguments
    parser.add_argument('--train', type=str, required=True,
                        help='Path to training CSV')
    parser.add_argument('--val', type=str, required=True,
                        help='Path to validation CSV')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test CSV')

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

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='baseline_outputs',
                        help='Output directory (default: baseline_outputs)')

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Run training
    trainer = BaselineTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

    """
    ================================================================================
BASELINE CODEBERT CLASSIFIER - WEEK 2 DELIVERABLE
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
  output_dir: baseline_outputs

================================================================================
INITIALIZING TOKENIZER
================================================================================
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.0/25.0 [00:00<00:00, 44.0kB/s]
vocab.json: 899kB [00:00, 4.66MB/s]
merges.txt: 456kB [00:00, 4.27MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:00<00:00, 773kB/s]
✓ CodeBERT tokenizer loaded

================================================================================
LOADING DATASETS
================================================================================

Train Dataset:
  Loaded 5947 samples from data/train.csv
  Positive samples: 3306
  Negative samples: 2641

Validation Dataset:
  Loaded 1274 samples from data/val.csv
  Positive samples: 708
  Negative samples: 566

Test Dataset:
  Loaded 1275 samples from data/test.csv
  Positive samples: 709
  Negative samples: 566

================================================================================
INITIALIZING MODEL
================================================================================
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 499M/499M [03:05<00:00, 2.69MB/s]
Model: BaselineCodeBERTClassifier
Total parameters: 124,843,010
Trainable parameters: 124,843,010
Device: cpu

================================================================================
TRAINING BASELINE CLASSIFIER
================================================================================

📌 STAGE 1: WARM-UP (Classifier only)
--------------------------------------------------------------------------------
Trainable parameters: 197,378

Warm-up Epoch 1/2
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 499M/499M [02:47<00:00, 2.97MB/s]
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 372/372 [37:54<00:00,  6.12s/it]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [03:17<00:00,  2.47s/it]
  Train Loss: 0.5631
  Val Loss: 0.4530
  Val Accuracy: 0.8501
  Val F1: 0.8781

Warm-up Epoch 2/2
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 372/372 [36:54<00:00,  5.95s/it]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [03:19<00:00,  2.49s/it]
  Train Loss: 0.4514
  Val Loss: 0.3828
  Val Accuracy: 0.8571
  Val F1: 0.8821

📌 STAGE 2: FINE-TUNING (All layers)
--------------------------------------------------------------------------------
Trainable parameters: 124,843,010

Fine-tune Epoch 1/5
Training:  47%|███████████████████████████████████████████████████████▏                                                              | 174/372 [1:03:03<1:16:42, 23.24s/it]Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 372/372 [2:16:27<00:00, 22.01s/it]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [02:52<00:00,  2.16s/it]
  Train Loss: 0.0599
  Val Loss: 0.0124
  Val Accuracy: 0.9969
  Val Precision: 0.9972
  Val Recall: 0.9972
  Val F1: 0.9972
  ✓ Saved best model (F1: 0.9972)

================================================================================
FINAL EVALUATION ON TEST SET
================================================================================
Loading best model from baseline_outputs/best_model.pt
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [03:05<00:00,  2.31s/it]

Test Results:
--------------------------------------------------------------------------------
Loss:      0.0160
Accuracy:  0.9961
Precision: 0.9972
Recall:    0.9958
F1 Score:  0.9965

Confusion Matrix:
  True Negatives:  564
  False Positives: 2
  False Negatives: 3
  True Positives:  706

✓ Results saved to baseline_outputs/test_results.json

================================================================================
BASELINE TRAINING COMPLETE!
================================================================================
Total time: 0:03:06.802817
Final Test F1: 0.9965
Models saved in: baseline_outputs
================================================================================

"""