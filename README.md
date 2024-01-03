# Sentiment Analysis on IMDB Dataset

## Project Overview
This project applies various machine learning models for sentiment analysis on the IMDB movie reviews dataset. It specifically explores custom transformer models, a TensorFlow neural network model, and a BERT-based uncased model to predict the sentiment of movie reviews as positive or negative.

## Custom Transformer Model
- Designed specifically for the IMDB dataset.
- Incorporates a self-attention mechanism for contextual understanding.
- Achieved around 88.72% accuracy in preliminary testing.

## TensorFlow Neural Network Model
- A simpler architecture using TensorFlow and Keras.
- Includes embedding, global average pooling, and dense layers.
- Reached 79% accuracy, showcasing its potential for basic sentiment analysis tasks.

## BERT-Based-Uncased Model
- Implemented using the 'Bert-base-uncased' model from Hugging Face's Transformers library.
- Fine-tuned for sentiment analysis, achieving a high validation accuracy of approximately 94%.
- Demonstrates superior contextual understanding due to deep pre-training.

## Installation
To run this project, ensure you have the following dependencies installed:

```plaintext
- Python 3.7 or above
- TensorFlow 2.x
- Hugging Face's Transformers library
- Other libraries as specified in the requirements.txt file
```

## Dataset
The IMDB dataset comprises movie reviews labeled as positive or negative. It is widely used for binary sentiment classification tasks.

## Performance Comparison
### TensorFlow Model: 
Suitable for basic sentiment analysis with a simpler architecture.
### Custom Transformer Model: 
Shows promising results; could achieve higher accuracy with further training and fine-tuning.
### BERT Model:
Best performing model with deep contextual understanding; ideal for complex NLP tasks.

## Future Work
Enhance the custom transformer's accuracy with extended training and hyperparameter tuning.
Explore additional NLP tasks like text summarization and machine translation using these models.

## References
Vaswani et al., "Attention is All You Need."
Zhuang et al., "A Survey on Efficient Training of Transformers."
Additional relevant literature.
Note: For the full implementati

## Presentation
For a live presentation, click [here](https://drive.google.com/file/d/1BqLSlUDxQvYDs2tzaU2Gy_IzNKLSyJGN/view?usp=drive_link)
