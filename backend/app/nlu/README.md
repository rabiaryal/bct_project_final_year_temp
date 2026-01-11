# NLU Model Training

This directory contains training scripts for both Intent Classification (BERT) and Named Entity Recognition (RoBERTa) models used in the college recommendation system.

## Setup

1. Install training dependencies:
```bash
pip install -r training_requirements.txt
```

2. Ensure you have the training data in the correct format (see Data Format section below).

## Training Scripts

### 1. Individual Model Training

#### Intent Classification (BERT)
```bash
python nlu/intent/train_bert_intent.py \
  --data_path "../../data/intent/intent_data.json" \
  --output_dir "../../models/new_bert_intent_model" \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 5e-5
```

#### Entity Recognition (RoBERTa)
```bash
python nlu/entity/train_roberta_ner.py \
  --data_path "../../data/entity/entity_training_data.csv" \
  --output_dir "../../models/new_roberta_entity_model" \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 5e-5
```

### 2. Unified Training (Both Models)

```bash
python nlu/train_models.py \
  --mode both \
  --intent_data "../../data/intent/intent_data.json" \
  --entity_data "../../data/entity/entity_training_data.csv" \
  --intent_output "../../models/new_bert_intent_model" \
  --entity_output "../../models/new_roberta_entity_model" \
  --epochs 5 \
  --batch_size 16
```

#### Training Only Intent Model
```bash
python nlu/train_models.py \
  --mode intent \
  --intent_data "../../data/intent/intent_data.json" \
  --intent_output "../../models/new_bert_intent_model" \
  --epochs 3
```

#### Training Only Entity Model
```bash
python nlu/train_models.py \
  --mode entity \
  --entity_data "../../data/entity/entity_training_data.csv" \
  --entity_output "../../models/new_roberta_entity_model" \
  --epochs 3
```

## Data Format

### Intent Data Format
The intent training data should be in JSON format:

**Option 1: List format**
```json
[
  {
    "text": "Tell me about Kathmandu University",
    "intent": "GET_COLLEGE_INFO"
  },
  {
    "text": "What courses are available at KU?",
    "intent": "GET_COURSE_INFO"
  }
]
```

**Option 2: Dictionary format**
```json
{
  "GET_COLLEGE_INFO": [
    "Tell me about Kathmandu University",
    "Information about KU",
    "Details of Kathmandu University"
  ],
  "GET_COURSE_INFO": [
    "What courses are available?",
    "Show me available programs"
  ]
}
```

### Entity Data Format

**CSV Format (sentence_id, token, label)**
```csv
sentence_id,token,label
1,Tell,O
1,me,O
1,about,O
1,Kathmandu,B-COLLEGE_NAME
1,University,I-COLLEGE_NAME
2,What,O
2,is,O
2,the,O
2,fee,B-FEE
```

**JSON Format**
```json
[
  {
    "text": "Tell me about Kathmandu University",
    "entities": [
      {
        "start": 14,
        "end": 33,
        "label": "COLLEGE_NAME"
      }
    ]
  }
]
```

## Model Architecture

### BERT Intent Classifier
- **Base Model**: `bert-base-uncased`
- **Task**: Sequence Classification
- **Output**: Intent labels (23 classes)
- **Metrics**: Accuracy, Precision, Recall, F1-score

### RoBERTa Entity Extractor
- **Base Model**: `roberta-base`
- **Task**: Token Classification (NER)
- **Output**: BIO tags for entity recognition
- **Labels**: COLLEGE_NAME, LOCATION, PROGRAM, FEE, FACILITY, etc.
- **Metrics**: Entity-level Precision, Recall, F1-score

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 3 | Number of training epochs |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 5e-5 | Learning rate for optimizer |
| `max_length` | 128 | Maximum sequence length |
| `warmup_steps` | 500 | Warmup steps for learning rate |
| `weight_decay` | 0.01 | Weight decay for regularization |

## Output Structure

After training, each model directory will contain:

```
model_output_dir/
├── config.json              # Model configuration
├── pytorch_model.bin         # Model weights
├── tokenizer.json           # Tokenizer configuration
├── vocab.txt               # Vocabulary (BERT)
├── merges.txt              # Merges (RoBERTa)
├── label_mapping.json      # Intent labels (Intent model)
├── label_mappings.json     # Entity labels (Entity model)
├── training_results.json   # Training metrics
├── classification_report.txt # Detailed evaluation
└── logs/                   # Training logs
    └── events.out.tfevents.*
```

## Evaluation Metrics

### Intent Classification
- **Accuracy**: Overall prediction accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-score**: Per-class and weighted average

### Entity Recognition
- **Entity-level F1**: Primary metric for NER
- **Precision**: Exact entity match precision
- **Recall**: Exact entity match recall
- **Token-level accuracy**: Token classification accuracy

## Monitoring Training

### TensorBoard
View training progress with TensorBoard:
```bash
tensorboard --logdir model_output_dir/logs
```

### Training Logs
Training progress is logged to console and saved in the model directory.

## Tips for Better Performance

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Class Balance**: Check for class imbalance in intent data
3. **Entity Consistency**: Use consistent BIO tagging for entities
4. **Validation**: Monitor validation metrics to avoid overfitting
5. **Hyperparameters**: Experiment with different learning rates and batch sizes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or max_length
2. **Poor Performance**: Increase training data or epochs
3. **Overfitting**: Add more validation data or reduce learning rate
4. **Label Mismatch**: Ensure consistent label format in training data

### Memory Requirements
- **Intent Training**: ~2GB GPU memory (batch_size=16)
- **Entity Training**: ~3GB GPU memory (batch_size=16)
- **CPU Training**: Supported but slower

## Next Steps

1. **Evaluate Models**: Use the test scripts in `test/` directory
2. **Deploy Models**: Copy trained models to `models/` directory
3. **Update Config**: Update model paths in `utils/config.py`
4. **Test Integration**: Run full system tests

## Example Training Session

```bash
# 1. Prepare environment
conda activate bctproject
pip install -r nlu/training_requirements.txt

# 2. Train both models
python nlu/train_models.py \
  --mode both \
  --intent_data "../../data/intent/intent_data.json" \
  --entity_data "../../data/entity/entity_training_data.csv" \
  --intent_output "../../models/bert_intent_model_v2" \
  --entity_output "../../models/roberta_entity_model_v2" \
  --epochs 5 \
  --batch_size 16

# 3. Test trained models
python ../../test/test_intent_model.py --model_path "../../models/bert_intent_model_v2"
python ../../test/test_entity_model.py --model_path "../../models/roberta_entity_model_v2"

# 4. Update system configuration
# Edit backend/app/utils/config.py to use new model paths
```