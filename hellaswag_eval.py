"""
HellaSwag Benchmark Evaluation Script
Evaluates a GLaMA model on the HellaSwag commonsense reasoning benchmark.
"""

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from model import GLaMA, ModelConfig
from torch.nn.attention.flex_attention import create_block_mask

class HellaSwagEvaluator:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the HellaSwag evaluator.

        Args:
            model_path: Path to model checkpoint (if None, uses untrained model)
            device: Device to run evaluation on
        """
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Initialize model
        config = ModelConfig()
        self.model = GLaMA(config).to(device)

        # Load checkpoint if provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")

        self.model.eval()
        print(f"Model loaded with {self.model.get_num_params:,} parameters")

    def load_dataset(self, split='validation'):
        """Load HellaSwag dataset from HuggingFace."""
        print(f"Loading HellaSwag {split} split...")
        dataset = load_dataset("Rowan/hellaswag", split=split)
        print(f"Loaded {len(dataset)} examples")
        return dataset

    def preprocess_text(self, text):
        """Clean and prepare text for tokenization."""
        text = text.strip()
        return text

    def compute_log_likelihood(self, context, continuation):
        """
        Compute the log-likelihood of a continuation given a context.

        Args:
            context: The context string
            continuation: The continuation string to score

        Returns:
            Average log-likelihood per token
        """
        # Tokenize context and continuation separately
        context_tokens = self.tokenizer.encode(self.preprocess_text(context))
        continuation_tokens = self.tokenizer.encode(self.preprocess_text(continuation))

        # Combine them
        full_tokens = context_tokens + continuation_tokens

        # Ensure we don't exceed max sequence length
        max_len = self.model.config.seq_len
        if len(full_tokens) > max_len:
            # Truncate from the left (keep recent context + all continuation)
            overflow = len(full_tokens) - max_len
            context_tokens = context_tokens[overflow:]
            full_tokens = context_tokens + continuation_tokens

        # Convert to tensor
        input_ids = torch.tensor([full_tokens], dtype=torch.long).to(self.device)

        # Create simple block mask for single document
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=1,
            H=None,
            Q_LEN=input_ids.size(1),
            KV_LEN=input_ids.size(1),
            device=self.device
        )

        # Forward pass
        with torch.no_grad():
            # Get embeddings
            x = self.model.in_emb(input_ids)

            # Pass through transformer blocks
            for block in self.model.blocks:
                x = block(x, (self.model.cos, self.model.sin), block_mask)

            x = self.model.norm(x)
            logits = self.model.lm_head(x)

        # Calculate log probabilities for continuation tokens only
        # Shift logits and labels for next-token prediction
        context_len = len(context_tokens)
        continuation_len = len(continuation_tokens)

        # Get logits for positions that predict continuation tokens
        # logits[0, i] predicts token at position i+1
        relevant_logits = logits[0, context_len-1:context_len+continuation_len-1]
        relevant_labels = torch.tensor(continuation_tokens, dtype=torch.long).to(self.device)

        # Compute log probabilities
        log_probs = F.log_softmax(relevant_logits, dim=-1)

        # Get log probability of actual tokens
        token_log_probs = log_probs[range(continuation_len), relevant_labels]

        # Return average log likelihood
        return token_log_probs.mean().item()

    def evaluate_example(self, example):
        """
        Evaluate a single HellaSwag example.

        Args:
            example: A HellaSwag example dict

        Returns:
            bool: True if prediction is correct
        """
        # Extract context and endings
        context = example['ctx']
        endings = example['endings']
        label = int(example['label'])

        # Compute log-likelihood for each ending
        log_likelihoods = []
        for ending in endings:
            ll = self.compute_log_likelihood(context, ending)
            log_likelihoods.append(ll)

        # Predict the ending with highest log-likelihood
        prediction = np.argmax(log_likelihoods)

        return prediction == label

    def evaluate(self, split='validation', max_examples=None):
        """
        Run full evaluation on HellaSwag dataset.

        Args:
            split: Dataset split to evaluate on
            max_examples: Maximum number of examples to evaluate (None for all)

        Returns:
            dict: Evaluation results
        """
        dataset = self.load_dataset(split)

        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        correct = 0
        total = 0

        print(f"\nEvaluating on {len(dataset)} examples...")

        for example in tqdm(dataset):
            is_correct = self.evaluate_example(example)
            if is_correct:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'split': split
        }

        return results

    def print_results(self, results):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("HellaSwag Evaluation Results")
        print("="*50)
        print(f"Split: {results['split']}")
        print(f"Total Examples: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*50)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate GLaMA model on HellaSwag')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to evaluate (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = HellaSwagEvaluator(
        model_path=args.model_path,
        device=args.device
    )

    # Run evaluation
    results = evaluator.evaluate(
        split=args.split,
        max_examples=args.max_examples
    )

    # Print results
    evaluator.print_results(results)

    return results


if __name__ == "__main__":
    main()
