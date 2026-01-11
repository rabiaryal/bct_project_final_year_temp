# NLU Model Testing Scripts

This directory contains interactive testing scripts for individual model components.

## Available Test Scripts

### 1. Intent Model Tester (`test_intent_model.py`)
Tests the BERT intent classification model individually.

**Usage:**
```bash
cd /Applications/development/ml\ learning/bct_final_year_project/test
conda activate bctproject
python test_intent_model.py
```

**Features:**
- Interactive text input from terminal
- Displays predicted intent with confidence scores
- Shows top 5 predictions
- Confidence assessment (High/Medium/Low)
- Model metadata information

### 2. Entity Model Tester (`test_entity_model.py`)
Tests the RoBERTa entity extraction model individually.

**Usage:**
```bash
cd /Applications/development/ml\ learning/bct_final_year_project/test
conda activate bctproject
python test_entity_model.py
```

**Features:**
- Interactive text input from terminal
- Extracts and displays all detected entities
- Shows entity-to-slot mapping
- Lists available entity types
- Provides example inputs

### 3. Complete NLU Pipeline Tester (`test_nlu_pipeline.py`)
Tests both intent classification and entity extraction together.

**Usage:**
```bash
cd /Applications/development/ml\ learning/bct_final_year_project/test
conda activate bctproject
python test_nlu_pipeline.py
```

**Features:**
- Complete NLU analysis (intent + entities)
- Shows combined results and analysis
- Suggests next dialogue actions
- Confidence and entity richness assessment
- Session tracking with timestamps

## Model Information

**Intent Model:** BERT (bert-base-uncased)
- **Location:** `/Applications/development/ml learning/bct_final_year_project/models/bert_intent_model`
- **Classes:** 23 intents
- **Purpose:** Classifies user intent (college info, admission, fees, etc.)

**Entity Model:** RoBERTa 
- **Location:** `/Applications/development/ml learning/bct_final_year_project/models/roberta_entity_model`
- **Labels:** 25 entity labels (B-/I- format)
- **Purpose:** Extracts entities (college names, locations, programs, etc.)

## Example Test Inputs

Try these example sentences to test the models:

### Intent Examples:
- "Tell me about Kathmandu University"
- "What are the admission requirements?"
- "How much does engineering cost?"
- "Where is the university located?"
- "Hello, I need help finding a college"

### Entity Examples:
- "Tell me about Kathmandu University engineering programs"
- "What programs does Tribhuvan University offer in Kathmandu?"
- "I want to study computer engineering at Pokhara University"
- "Show me colleges with hostel facilities and scholarships"

## Troubleshooting

If you encounter issues:

1. **Import errors:** Make sure you're running from the test directory and the backend path is correct
2. **Model loading errors:** Verify the model files exist in the models directory
3. **CUDA/MPS errors:** The models will automatically fall back to CPU if GPU is unavailable
4. **torchvision warnings:** These can be safely ignored as we don't use image functionality

## Output Interpretation

### Intent Classification:
- **Confidence > 0.8:** High confidence, reliable prediction
- **Confidence 0.6-0.8:** Medium confidence, mostly reliable
- **Confidence < 0.6:** Low confidence, may need clarification

### Entity Extraction:
- **Entity types:** COLLEGE_NAME, LOCATION, PROGRAM, FACILITY, FEE, etc.
- **Slot mapping:** How entities map to dialogue slots
- **Coverage:** Number of entities found indicates input complexity