# Multimodal Product Classifier

This project implements a multimodal machine learning system for large-scale product classification by combining image features, text features, and categorical metadata using LightGBM.

It was developed for the Amazon ML Challenge and focuses on building a production-style pipeline for handling heterogeneous e-commerce data.

---

## Overview

Real-world product data often contains multiple modalities:

- Product images  
- Text descriptions and titles  
- Structured categorical metadata  

This project fuses all three modalities into a single supervised learning model to improve prediction performance and robustness.

---

## Architecture

The pipeline consists of:

### 1. Image Feature Extraction
- Pretrained CNN (EfficientNetB0)
- `include_top = False`
- Global Average Pooling
- Output: fixed-length image embeddings (1280-D)

### 2. Text Feature Extraction
- NLP vectorization (TF-IDF(ngram-(1,2)) / embedding based)
- Preprocessing: cleaning, tokenization

### 3. Metadata Handling
- Native categorical features
- preprocessing: skewed data-conversion, missing value imputation, null-value handling
- Handled directly using LightGBM categorical support

### 4. Feature Fusion
- Concatenation of:
  - Image embeddings  
  - Text vectors  
  - Categorical features  

### 5. Final Model
- LightGBM classifier / regressor
- Optimized for large-scale tabular + multimodal data

---

## Model Choice

LightGBM was selected due to:

- Native categorical feature handling  
- High performance on tabular data  
- Low memory footprint  
- Fast training and inference  
- Strong performance on large datasets  

---

## Dataset

The original dataset was provided as part of the Amazon ML Challenge and contains:

- Product images  
- Text fields (title, description, etc.)  
- Categorical metadata  
- Target labels

> Note: The dataset is not included in this repository due to size constraints and competition licensing restrictions.

---

## Training Pipeline

1. Preprocess images and extract CNN embeddings  
2. Preprocess text and generate feature vectors  
3. Encode categorical metadata  
4. Align features by product ID  
5. Train LightGBM model  
6. Evaluate using validation split  
7. Export trained model  
8. Make predictions on test set
9. Submit predictions to competition platform

---

## Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE) on log-transformed targets  
- Inverse transformation applied during evaluation:

```python
mae_price = np.exp(mae_log) - 1
```
This ensures predictions are evaluated in the original price scale.

---

## Results
- Achieved MAE (log-transformed):**0.517287**
- corresponding MAE (price scale):**0.67747**

## Limitations
- Dataset not publicly distributable
- Training requires high memory for feature matrices
- CNN backbone frozen (no fine-tuning)
- Evaluation limited to challenge metrics
### Image Backbone Experiments

Two CNN backbones were evaluated for image feature extraction:

- **EfficientNet-B0** (baseline)
  - Faster and lightweight
  - Used in early experiments

- **EfficientNet-B3** (final model)
  - Higher representational capacity
  - Better feature quality for downstream LightGBM fusion
  - Selected as the final image encoder

The final system uses EfficientNet-B3 embeddings for training and inference.

## Future Improvements
- End-to-end fine-tuning of image backbone
- Transformer-based text embeddings
- Model ensembling
- Online inference API
- Feature store integration
- Hyperparameter optimization

## License
This project is released under the Apache 2.0 License.

## Author
Developed by AJ.