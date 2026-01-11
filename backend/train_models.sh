#!/bin/bash

# Example Training Script for College Recommendation System Models
# Make sure you're in the backend directory and have activated the bctproject environment

echo "🚀 Starting NLU Model Training for College Recommendation System"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "app/nlu/train_models.py" ]; then
    echo "❌ Error: Please run this script from the backend/ directory"
    exit 1
fi

# Activate conda environment
echo "📦 Activating bctproject environment..."
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate bctproject

# Install training dependencies
echo "📚 Installing training dependencies..."
pip install -q transformers datasets scikit-learn seqeval accelerate tensorboard

# Create output directories
echo "📁 Creating output directories..."
mkdir -p ../models/bert_intent_model_retrained
mkdir -p ../models/roberta_entity_model_retrained

# Check if data files exist
INTENT_DATA="../data/intent/intent_data.json"
ENTITY_DATA="../data/entity/entity_training_data.csv"

if [ ! -f "$INTENT_DATA" ]; then
    echo "❌ Error: Intent data file not found at $INTENT_DATA"
    exit 1
fi

if [ ! -f "$ENTITY_DATA" ]; then
    echo "❌ Error: Entity data file not found at $ENTITY_DATA"
    exit 1
fi

echo "✅ Data files found:"
echo "   Intent data: $INTENT_DATA"
echo "   Entity data: $ENTITY_DATA"

# Training parameters
EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=5e-5
MAX_LENGTH=128

echo "⚙️  Training Parameters:"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Max Length: $MAX_LENGTH"

echo ""
echo "🎯 Starting model training..."

# Train both models
python app/nlu/train_models.py \
    --mode both \
    --intent_data "$INTENT_DATA" \
    --entity_data "$ENTITY_DATA" \
    --intent_output "../models/bert_intent_model_retrained" \
    --entity_output "../models/roberta_entity_model_retrained" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📊 Models saved to:"
    echo "   Intent model: ../models/bert_intent_model_retrained"
    echo "   Entity model: ../models/roberta_entity_model_retrained"
    echo ""
    echo "🔧 To use these models, update the paths in app/utils/config.py:"
    echo "   intent_model_path: '../models/bert_intent_model_retrained'"
    echo "   entity_model_path: '../models/roberta_entity_model_retrained'"
else
    echo "❌ Training failed! Check the output above for errors."
    exit 1
fi