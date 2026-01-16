"""
Utility functions for the NER model.
"""
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json
from sklearn.metrics import confusion_matrix
import itertools

def merge_subwords(tokens, labels):
    """Merge subword tokens and their labels."""
    merged_tokens = []
    merged_labels = []
    
    for token, label in zip(tokens, labels):
        if token.startswith('Ġ'):
            merged_tokens.append(token[1:])
            merged_labels.append(label)
        elif merged_tokens:
            merged_tokens[-1] += token
        else:
            # Handle cases where the first token is not a start token
            merged_tokens.append(token)
            merged_labels.append(label)
            
    return merged_tokens, merged_labels

def post_process_entities(tokens, labels):
    """Post-process entities to merge BIO tags and clean up, handling RoBERTa's subwords."""
    if not tokens:
        return []

    # Step 1: Merge subword tokens based on RoBERTa's 'Ġ' prefix.
    merged_tokens = []
    merged_labels = []
    for token, label in zip(tokens, labels):
        if token.startswith('Ġ'):
            # Start of a new word
            merged_tokens.append(token[1:])
            merged_labels.append(label)
        elif merged_tokens:
            # Continuation of the previous word
            merged_tokens[-1] += token
        else:
            # First token doesn't have a 'Ġ' prefix (unusual but handle it)
            merged_tokens.append(token)
            merged_labels.append(label)

    # Step 2: Extract entities from merged tokens using BIO logic.
    entities = []
    current_entity_tokens = []
    current_entity_label = None

    for token, label in zip(merged_tokens, merged_labels):
        if label.startswith('B-'):
            # If there's a current entity, save it before starting a new one.
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append({'value': entity_text, 'type': current_entity_label})

            # Start a new entity.
            current_entity_tokens = [token]
            current_entity_label = label[2:]
        elif label.startswith('I-') and current_entity_label == label[2:]:
            # Continue the current entity.
            current_entity_tokens.append(token)
        else:
            # Not an entity token or a different entity, so close the current one.
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append({'value': entity_text, 'type': current_entity_label})
            current_entity_tokens = []
            current_entity_label = None

    # Add the last entity if it exists.
    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entities.append({'value': entity_text, 'type': current_entity_label})

    # Step 3: Post-processing filters (optional but recommended).
    final_entities = []
    for entity in entities:
        text = entity['value'].strip()
        if text and len(text) >= 1:  # Keep entities with at least 1 character
            final_entities.append(entity)

    return final_entities

def post_process_entities_with_confidence(tokens, labels, scores):
    """
    Post-process entities with confidence scores (entity-level thresholding safe).
    
    Args:
        tokens: List of tokens from tokenizer
        labels: List of predicted BIO labels  
        scores: List of confidence scores per token
    
    Returns:
        List of entities with confidence scores calculated at entity level
    """
    if not tokens:
        return []

    # Step 1: Merge subword tokens based on RoBERTa's 'Ġ' prefix
    merged_tokens = []
    merged_labels = []
    merged_scores = []
    
    for token, label, score in zip(tokens, labels, scores):
        if token.startswith('Ġ'):
            # Start of a new word
            merged_tokens.append(token[1:])
            merged_labels.append(label)
            merged_scores.append(score)
        elif merged_tokens:
            # Continuation of the previous word - merge with previous
            merged_tokens[-1] += token
            # Keep the previous label (don't change it)
            # Average the scores for merged tokens
            merged_scores[-1] = (merged_scores[-1] + score) / 2
        else:
            # First token doesn't have a 'Ġ' prefix (unusual but handle it)
            merged_tokens.append(token)
            merged_labels.append(label)
            merged_scores.append(score)

    # Step 2: Extract entities from merged tokens using BIO logic
    entities = []
    current_entity_tokens = []
    current_entity_label = None
    current_entity_scores = []

    for token, label, score in zip(merged_tokens, merged_labels, merged_scores):
        if label.startswith('B-'):
            # If there's a current entity, save it before starting a new one
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                # Calculate entity confidence as average of token scores
                entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                entities.append({
                    'text': entity_text, 
                    'type': current_entity_label,
                    'confidence': entity_confidence
                })

            # Start a new entity
            current_entity_tokens = [token]
            current_entity_label = label[2:]  # Remove B- prefix
            current_entity_scores = [score]
            
        elif label.startswith('I-') and current_entity_label == label[2:]:
            # Continue the current entity
            current_entity_tokens.append(token)
            current_entity_scores.append(score)
            
        else:
            # Not an entity token or different entity - close current one
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                entities.append({
                    'text': entity_text,
                    'type': current_entity_label,
                    'confidence': entity_confidence
                })
            current_entity_tokens = []
            current_entity_label = None
            current_entity_scores = []

    # Add the last entity if it exists
    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
        entities.append({
            'text': entity_text,
            'type': current_entity_label, 
            'confidence': entity_confidence
        })

    return entities
    for entity in entities:
        # Filter out very short or punctuation-only entities.
        if len(entity['value'].strip()) >= 3 and not all(not c.isalnum() for c in entity['value']):
            final_entities.append(entity)
            
    return final_entities

def compute_metrics(eval_pred, id_to_label):
    """Compute NER metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
        'accuracy': accuracy_score(true_labels, true_predictions),
    }

def evaluate(test_dataset, trainer, id_to_label):
    """Evaluate the trained model with comprehensive metrics"""
    print("Evaluating model...")
    results = trainer.evaluate(test_dataset)
    
    # Detailed predictions for classification report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=2)
    y_true = predictions.label_ids
    
    # Convert to label names
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(y_pred, y_true):
        pred_labels = []
        true_labels_seq = []
        for p, l in zip(prediction, label):
            if l != -100:
                pred_labels.append(id_to_label[p])
                true_labels_seq.append(id_to_label[l])
        true_predictions.append(pred_labels)
        true_labels.append(true_labels_seq)
    
    # Generate classification report
    from seqeval.metrics import classification_report as seq_classification_report
    report = seq_classification_report(true_labels, true_predictions)
    
    print("\nClassification Report:")
    print(report)
    
    # Calculate additional metrics
    evaluation_metrics = _calculate_detailed_metrics(true_labels, true_predictions)
    
    return results, report, evaluation_metrics, true_labels, true_predictions

def _calculate_detailed_metrics(true_labels, predicted_labels):
    """Calculate detailed entity-level metrics"""
    from seqeval.metrics import precision_score, recall_score, f1_score
    from seqeval.metrics import classification_report as seq_classification_report
    
    # Entity-level metrics
    entity_precision = precision_score(true_labels, predicted_labels)
    entity_recall = recall_score(true_labels, predicted_labels)
    entity_f1 = f1_score(true_labels, predicted_labels)
    
    # Per-entity F1 scores
    report_dict = seq_classification_report(true_labels, predicted_labels, output_dict=True)
    
    # BIO validity check
    bio_validity = _check_bio_validity(predicted_labels)
    
    # Flatten labels for confusion matrix
    flat_true = [label for seq in true_labels for label in seq]
    flat_pred = [label for seq in predicted_labels for label in seq]
    
    metrics = {
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'entity_f1': entity_f1,
        'per_entity_metrics': report_dict,
        'bio_validity_rate': bio_validity,
        'flat_true_labels': flat_true,
        'flat_predicted_labels': flat_pred
    }
    
    return metrics

def _check_bio_validity(predicted_sequences):
    """Check BIO tagging validity"""
    valid_sequences = 0
    total_sequences = len(predicted_sequences)
    
    for sequence in predicted_sequences:
        is_valid = True
        prev_tag = 'O'
        
        for tag in sequence:
            if tag.startswith('I-'):
                entity_type = tag[2:]
                # I- tag should follow B- or I- of same entity type
                if prev_tag not in [f'B-{entity_type}', f'I-{entity_type}']:
                    is_valid = False
                    break
            prev_tag = tag
        
        if is_valid:
            valid_sequences += 1
    
    return valid_sequences / total_sequences if total_sequences > 0 else 0.0

def create_visualizations(graphics_dir, texts, labels, evaluation_metrics=None, trainer_logs=None):
    """Create comprehensive training and evaluation visualizations"""
    print(f"Creating visualizations in {graphics_dir}")
    
    # 1. Label distribution
    all_labels = [label for seq in labels for label in seq if label != 'O']
    label_counts = Counter(all_labels)
    
    plt.figure(figsize=(14, 8))
    labels_list = list(label_counts.keys())
    counts = list(label_counts.values())
    
    plt.bar(labels_list, counts)
    plt.title('Entity Label Distribution', fontsize=16)
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sentence length distribution
    sentence_lengths = [len(text.split() if isinstance(text, str) else text) for text in texts]
    
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Sentence Length Distribution', fontsize=16)
    plt.xlabel('Sentence Length (tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(np.mean(sentence_lengths), color='red', linestyle='--', 
               label=f'Mean: {np.mean(sentence_lengths):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'sentence_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Entities per sentence distribution
    entities_per_sentence = []
    for seq in labels:
        entity_count = sum(1 for label in seq if label.startswith('B-'))
        entities_per_sentence.append(entity_count)
    
    plt.figure(figsize=(10, 6))
    plt.hist(entities_per_sentence, bins=range(max(entities_per_sentence) + 2), 
            alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Entities per Sentence Distribution', fontsize=16)
    plt.xlabel('Number of Entities', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'entities_per_sentence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4-8. Advanced evaluation visualizations (if metrics provided)
    if evaluation_metrics:
        _create_evaluation_charts(graphics_dir, evaluation_metrics)
    
    # 9. Training loss curve (if trainer logs provided)
    if trainer_logs:
        _create_loss_curve(graphics_dir, trainer_logs)
    
    print(f"✅ All visualizations saved to {graphics_dir}")

def _create_evaluation_charts(graphics_dir, metrics):
    """Create evaluation-specific charts"""
    
    # 4. Entity-level metrics bar chart
    entity_metrics = [
        metrics['entity_precision'], 
        metrics['entity_recall'], 
        metrics['entity_f1']
    ]
    metric_names = ['Precision', 'Recall', 'F1-Score']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, entity_metrics, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Entity-level Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, entity_metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'entity_level_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Per-entity F1 table visualization
    if 'per_entity_metrics' in metrics:
        entity_f1_scores = {}
        for entity, scores in metrics['per_entity_metrics'].items():
            if entity not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg'] and isinstance(scores, dict):
                entity_f1_scores[entity] = scores.get('f1-score', 0)
        
        if entity_f1_scores:
            plt.figure(figsize=(12, 8))
            entities = list(entity_f1_scores.keys())
            f1_scores = list(entity_f1_scores.values())
            
            bars = plt.bar(entities, f1_scores, color='lightcoral')
            plt.title('Per-Entity F1 Scores', fontsize=16)
            plt.xlabel('Entity Types', fontsize=12)
            plt.ylabel('F1-Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(graphics_dir, 'per_entity_f1_scores.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. BIO validity visualization
    validity_rate = metrics.get('bio_validity_rate', 0)
    invalid_rate = 1 - validity_rate
    
    plt.figure(figsize=(8, 8))
    sizes = [validity_rate, invalid_rate]
    labels = [f'Valid BIO ({validity_rate:.1%})', f'Invalid BIO ({invalid_rate:.1%})']
    colors = ['#66BB6A', '#FF7043']
    
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 12})
    plt.title('BIO Tagging Validity Rate', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'bio_validity_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Confusion Matrix
    flat_true = metrics['flat_true_labels']
    flat_pred = metrics['flat_predicted_labels']
    
    # Get unique labels
    unique_labels = sorted(list(set(flat_true + flat_pred)))
    cm = confusion_matrix(flat_true, flat_pred, labels=unique_labels)
    
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Entity Recognition', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, unique_labels)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def _create_loss_curve(graphics_dir, trainer_logs):
    """Create comprehensive training visualizations"""
    # Extract training metrics from logs
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_accuracy = []
    eval_f1 = []
    eval_epochs = []
    learning_rates = []
    
    for log in trainer_logs:
        if 'loss' in log and 'eval_loss' not in log:  # Training loss
            train_losses.append(log['loss'])
            train_steps.append(log.get('step', len(train_losses)))
            if 'learning_rate' in log:
                learning_rates.append(log['learning_rate'])
        elif 'eval_loss' in log:  # Evaluation metrics
            eval_losses.append(log['eval_loss'])
            eval_accuracy.append(log.get('eval_accuracy', 0))
            eval_f1.append(log.get('eval_f1', 0))
            eval_epochs.append(log.get('epoch', len(eval_losses)))
    
    # 1. Training vs Validation Loss
    if train_losses or eval_losses:
        plt.figure(figsize=(12, 8))
        
        if train_losses and train_steps:
            plt.plot(train_steps, train_losses, label='Training Loss', linewidth=2, color='#FF6B6B', alpha=0.8)
        
        if eval_losses and eval_epochs:
            # Convert epochs to approximate steps for alignment
            if train_steps:
                steps_per_epoch = max(train_steps) / max(eval_epochs) if eval_epochs else 1
                eval_steps_approx = [epoch * steps_per_epoch for epoch in eval_epochs]
                plt.plot(eval_steps_approx, eval_losses, 'o-', label='Validation Loss', 
                        linewidth=3, color='#4ECDC4', markersize=8, markerfacecolor='white', 
                        markeredgewidth=2)
            
        plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'training_validation_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Training vs Validation Accuracy
    if eval_accuracy and eval_epochs:
        plt.figure(figsize=(12, 8))
        if train_steps:
            steps_per_epoch = max(train_steps) / max(eval_epochs) if eval_epochs else 1
            eval_steps_approx = [epoch * steps_per_epoch for epoch in eval_epochs]
            plt.plot(eval_steps_approx, eval_accuracy, 'o-', label='Validation Accuracy', 
                    linewidth=3, color='#45B7D1', markersize=8, markerfacecolor='white', 
                    markeredgewidth=2)
        else:
            plt.plot(eval_epochs, eval_accuracy, 'o-', label='Validation Accuracy', 
                    linewidth=3, color='#45B7D1', markersize=8)
        
        plt.title('Validation Accuracy Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps' if train_steps else 'Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels on points
        for i, (x, y) in enumerate(zip(eval_steps_approx if train_steps else eval_epochs, eval_accuracy)):
            plt.annotate(f'{y:.1%}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'validation_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Training vs Validation F1 Score
    if eval_f1 and eval_epochs:
        plt.figure(figsize=(12, 8))
        if train_steps:
            steps_per_epoch = max(train_steps) / max(eval_epochs) if eval_epochs else 1
            eval_steps_approx = [epoch * steps_per_epoch for epoch in eval_epochs]
            plt.plot(eval_steps_approx, eval_f1, 'o-', label='Validation F1 Score', 
                    linewidth=3, color='#96CEB4', markersize=8, markerfacecolor='white', 
                    markeredgewidth=2)
        else:
            plt.plot(eval_epochs, eval_f1, 'o-', label='Validation F1 Score', 
                    linewidth=3, color='#96CEB4', markersize=8)
        
        plt.title('Validation F1 Score Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps' if train_steps else 'Epochs', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels on points
        for i, (x, y) in enumerate(zip(eval_steps_approx if train_steps else eval_epochs, eval_f1)):
            plt.annotate(f'{y:.1%}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'validation_f1_score.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Learning Rate Schedule
    if learning_rates and train_steps:
        plt.figure(figsize=(12, 8))
        plt.plot(train_steps, learning_rates, linewidth=2, color='#FFA07A', marker='o', markersize=4)
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')  # Log scale for learning rate
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Combined Metrics Dashboard
    if eval_losses and eval_accuracy and eval_f1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss subplot
        if train_steps:
            steps_per_epoch = max(train_steps) / max(eval_epochs) if eval_epochs else 1
            eval_steps_approx = [epoch * steps_per_epoch for epoch in eval_epochs]
            
            if train_losses:
                ax1.plot(train_steps, train_losses, label='Training Loss', color='#FF6B6B', alpha=0.8)
            ax1.plot(eval_steps_approx, eval_losses, 'o-', label='Validation Loss', 
                    color='#4ECDC4', markersize=6)
        else:
            ax1.plot(eval_epochs, eval_losses, 'o-', label='Validation Loss', color='#4ECDC4')
        
        ax1.set_title('Loss', fontweight='bold')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy subplot
        x_vals = eval_steps_approx if train_steps else eval_epochs
        ax2.plot(x_vals, eval_accuracy, 'o-', color='#45B7D1', markersize=6)
        ax2.set_title('Accuracy', fontweight='bold')
        ax2.set_xlabel('Steps' if train_steps else 'Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # F1 Score subplot
        ax3.plot(x_vals, eval_f1, 'o-', color='#96CEB4', markersize=6)
        ax3.set_title('F1 Score', fontweight='bold')
        ax3.set_xlabel('Steps' if train_steps else 'Epochs')
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate subplot
        if learning_rates and train_steps:
            ax4.plot(train_steps, learning_rates, color='#FFA07A', marker='o', markersize=4)
            ax4.set_title('Learning Rate', fontweight='bold')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nData Unavailable', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Learning Rate', fontweight='bold')
        
        plt.suptitle('Training Metrics Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(graphics_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Training visualization charts saved:")
    print(f"  ✓ Training vs Validation Loss")
    print(f"  ✓ Validation Accuracy Over Time") 
    print(f"  ✓ Validation F1 Score Over Time")
    print(f"  ✓ Learning Rate Schedule")
    print(f"  ✓ Combined Training Dashboard")
