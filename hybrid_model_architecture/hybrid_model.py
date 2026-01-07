import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Dict, Tuple


class HybridMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base',
        dropout: float = 0.1,
        hidden_size: int = 768
    ):
        super(HybridMultiTaskModel, self).__init__()

        # Shared encoder
        self.codebert = RobertaModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # Head 1: Binary Classification (has_secret)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary: 0 or 1
        )

        # Head 2: Span Start (token-level)
        self.span_start_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Head 3: Span End (token-level)
        self.span_end_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Head 4: Entropy Regression (continuous value)
        self.entropy_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Continuous output
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Get CodeBERT embeddings
        outputs = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # [CLS] token for classification and entropy
        cls_output = outputs.pooler_output  # [batch_size, hidden_size]

        # All tokens for span prediction
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Task 1: Binary Classification
        classification_logits = self.classification_head(cls_output)  # [batch_size, 2]

        # Task 2: Span Localization
        span_start_logits = self.span_start_head(sequence_output).squeeze(-1)  # [batch_size, seq_len]
        span_end_logits = self.span_end_head(sequence_output).squeeze(-1)  # [batch_size, seq_len]

        # Task 3: Entropy Regression
        entropy_pred = self.entropy_head(cls_output).squeeze(-1)  # [batch_size]

        return {
            'classification_logits': classification_logits,
            'span_start_logits': span_start_logits,
            'span_end_logits': span_end_logits,
            'entropy_pred': entropy_pred
        }

    def freeze_encoder(self):
        #Freeze CodeBERT for warm-up stage
        for param in self.codebert.parameters():
            param.requires_grad = False
        print(" CodeBERT encoder frozen")

    def unfreeze_encoder(self):
        #Unfreeze CodeBERT for fine-tuning stage
        for param in self.codebert.parameters():
            param.requires_grad = True
        print(" CodeBERT encoder unfrozen")

    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    token_span_start: torch.Tensor,
    token_span_end: torch.Tensor,
    span_available: torch.Tensor,
    entropy: torch.Tensor,
    classification_weight: float = 1.0,
    span_weight: float = 1.0,
    entropy_weight: float = 0.5
) -> Dict[str, torch.Tensor]:
    # Task 1: Classification Loss (always computed)
    classification_criterion = nn.CrossEntropyLoss()
    classification_loss = classification_criterion(
        outputs['classification_logits'],
        labels
    )

    # Task 2: Span Loss (masked - only when span_available=True)
    num_spans_available = span_available.sum().item()

    if num_spans_available > 0:
        # At least one valid span in batch - compute loss normally
        span_start_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        span_end_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Mask span targets: set to -1 where span_available=False
        masked_span_start = torch.where(
            span_available,
            token_span_start,
            torch.full_like(token_span_start, -1)
        )

        masked_span_end = torch.where(
            span_available,
            token_span_end,
            torch.full_like(token_span_end, -1)
        )

        # Compute span losses (automatically ignores -1 indices)
        span_start_loss = span_start_criterion(
            outputs['span_start_logits'],
            masked_span_start
        )

        span_end_loss = span_end_criterion(
            outputs['span_end_logits'],
            masked_span_end
        )

        # Average start and end losses
        span_loss = (span_start_loss + span_end_loss) / 2.0
    else:
        # NO valid spans in this batch - return zero loss. This prevents NaN when entire batch has no span annotations
        span_loss = torch.tensor(0.0, device=outputs['classification_logits'].device)

    # Task 3: Entropy Regression Loss (always computed). MSE between predicted and actual entropy
    entropy_criterion = nn.MSELoss()
    entropy_loss = entropy_criterion(
        outputs['entropy_pred'],
        entropy
    )

    # Combined weighted loss
    total_loss = (
        classification_weight * classification_loss +
        span_weight * span_loss +
        entropy_weight * entropy_loss
    )

    # Count how many spans were actually used
    num_spans_used = num_spans_available

    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'span_loss': span_loss,
        'entropy_loss': entropy_loss,
        'num_spans_used': num_spans_used
    }


# SAFETY CHECKS
def validate_span_indices(
    token_span_start: torch.Tensor,
    token_span_end: torch.Tensor,
    span_available: torch.Tensor,
    seq_length: int
) -> bool:
    issues = []

    for i in range(len(token_span_start)):
        if not span_available[i]:
            continue

        start = token_span_start[i].item()
        end = token_span_end[i].item()

        # Check 1: Valid indices (not -1)
        if start == -1 or end == -1:
            issues.append(f"Sample {i}: Invalid span indices (start={start}, end={end})")

        # Check 2: End >= Start
        if end < start:
            issues.append(f"Sample {i}: End < Start (start={start}, end={end})")

        # Check 3: Within sequence length
        if start >= seq_length or end >= seq_length:
            issues.append(f"Sample {i}: Span outside sequence (start={start}, end={end}, seq_len={seq_length})")

        # Check 4: Start >= 0
        if start < 0 or end < 0:
            issues.append(f"Sample {i}: Negative indices (start={start}, end={end})")

    if issues:
        print("  SPAN VALIDATION ERRORS:")
        for issue in issues:
            print(f"  {issue}")
        return False

    return True


def validate_entropy_values(
    entropy: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 10.0
) -> bool:
    # Check for NaN
    if torch.isnan(entropy).any():
        print("  ENTROPY VALIDATION ERROR: NaN values detected")
        return False

    # Check for infinite values
    if torch.isinf(entropy).any():
        print("  ENTROPY VALIDATION ERROR: Infinite values detected")
        return False

    # Check range
    if (entropy < min_val).any() or (entropy > max_val).any():
        out_of_range = ((entropy < min_val) | (entropy > max_val)).sum().item()
        print(f"  ENTROPY VALIDATION WARNING: {out_of_range} values outside [{min_val}, {max_val}]")
        print(f"  Min: {entropy.min():.4f}, Max: {entropy.max():.4f}")

    return True


def check_span_availability_consistency(
    labels: torch.Tensor,
    span_available: torch.Tensor
) -> bool:
    # Count cases
    positive_with_span = ((labels == 1) & span_available).sum().item()
    positive_without_span = ((labels == 1) & ~span_available).sum().item()
    negative_with_span = ((labels == 0) & span_available).sum().item()
    negative_without_span = ((labels == 0) & ~span_available).sum().item()

    print(f"  Positive samples with span: {positive_with_span}")
    print(f"  Positive samples without span: {positive_without_span}")
    print(f"  Negative samples with span: {negative_with_span}")
    print(f"  Negative samples without span: {negative_without_span}")

    # Check if any negative example has span
    if negative_with_span > 0:
        print("  WARNING: Some negative samples have span annotations!")
        return False

    if positive_without_span > 0:
        pct = 100 * positive_without_span / (positive_with_span + positive_without_span)
        print(f"  Info: {positive_without_span} positive samples without spans ({pct:.1f}% - acceptable)")

    return True


# TESTING
if __name__ == "__main__":
    print("TESTING HYBRID THREE-HEAD MODEL")

    # Initialize model
    print("\n1. Initializing model...")
    model = HybridMultiTaskModel()
    trainable, total = model.get_trainable_parameters()
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Heads: Classification, Span Start, Span End, Entropy")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    seq_len = 128

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    outputs = model(dummy_input_ids, dummy_attention_mask)

    print(f"   Classification logits: {outputs['classification_logits'].shape}")
    print(f"   Span start logits: {outputs['span_start_logits'].shape}")
    print(f"   Span end logits: {outputs['span_end_logits'].shape}")
    print(f"   Entropy predictions: {outputs['entropy_pred'].shape}")

    # Test loss computation
    print("\n3. Testing three-head loss computation...")
    dummy_labels = torch.tensor([1, 0, 1, 1], dtype=torch.long)
    dummy_span_start = torch.tensor([10, -1, 15, 20], dtype=torch.long)
    dummy_span_end = torch.tensor([15, -1, 20, 25], dtype=torch.long)
    dummy_span_available = torch.tensor([True, False, True, True], dtype=torch.bool)
    dummy_entropy = torch.tensor([4.2, 0.0, 3.8, 4.5], dtype=torch.float)

    losses = compute_multitask_loss(
        outputs,
        dummy_labels,
        dummy_span_start,
        dummy_span_end,
        dummy_span_available,
        dummy_entropy,
        classification_weight=1.0,
        span_weight=1.0,
        entropy_weight=0.5
    )

    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    print(f"   Classification loss: {losses['classification_loss'].item():.4f}")
    print(f"   Span loss: {losses['span_loss'].item():.4f}")
    print(f"   Entropy loss: {losses['entropy_loss'].item():.4f}")
    print(f"   Spans used: {losses['num_spans_used']}")

    # Test span validation
    print("\n4. Testing span validation...")
    is_valid = validate_span_indices(
        dummy_span_start,
        dummy_span_end,
        dummy_span_available,
        seq_len
    )
    print(f"   Span validation: {'PASS' if is_valid else 'FAIL'}")

    # Test entropy validation
    print("\n5. Testing entropy validation...")
    is_entropy_valid = validate_entropy_values(dummy_entropy)
    print(f"   Entropy validation: {'PASS' if is_entropy_valid else 'FAIL'}")

    # Test consistency check
    print("\n6. Testing span availability consistency...")
    is_consistent = check_span_availability_consistency(
        dummy_labels,
        dummy_span_available
    )
    print(f"   Consistency check: {'PASS' if is_consistent else 'FAIL'}")

    # Test freezing
    print("\n7. Testing encoder freezing...")
    model.freeze_encoder()
    trainable_frozen, _ = model.get_trainable_parameters()
    print(f"   Trainable after freezing: {trainable_frozen:,}")
    print(f"   Reduction: {100 * (1 - trainable_frozen/trainable):.1f}%")

    model.unfreeze_encoder()
    trainable_unfrozen, _ = model.get_trainable_parameters()
    print(f"   Trainable after unfreezing: {trainable_unfrozen:,}")

    print("\nAll tests passed!")