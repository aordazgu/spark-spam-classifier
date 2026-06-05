# Spark Spam Classifier

**Course:** CSC369 (Big Data Fundamentals)

**Author:** Arturo Ordaz-Gutierrez

## Overview

This project builds a distributed email spam classifier using Apache Spark and k-NN in Scala. It compares two text feature extraction approaches, Bag of Words and TF-IDF, on a dataset of 8,000 real emails to determine which representation better captures spam-indicative patterns. The system demonstrates practical applications of distributed computing, NLP preprocessing, and machine learning evaluation.

## Research Question

**Primary:** Can TF-IDF feature weighting improve spam classification performance compared to simple word frequency counts? How significant is the improvement in a distributed setting?

**Secondary:** Which text representation more effectively captures spam-indicative language patterns?

## Methods

- **Dataset:** 8,000 emails (3,779 spam, 4,221 ham) from real-world sources
- **Preprocessing:** Text cleaning, tokenization, email-specific stopword removal (HTML tags, URLs, web artifacts)
- **Feature Extraction:** 
  - Bag of Words (baseline: raw word counts)
  - TF-IDF (term frequency weighted by inverse document frequency)
- **Classification:** k-NN with k=5, cosine similarity distance metric
- **Train/Test Split:** 80/20 stratified split (6,400 train, 1,600 test)
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, confusion matrix

## Key Findings

- **TF-IDF significantly outperforms Bag of Words:** 90.74% vs 70.73% accuracy, a 20-point improvement
- **TF-IDF provides superior precision and recall:** 90.26% precision and 89.27% recall vs 66.09% and 73.18% respectively
- **Feature engineering matters:** IDF weighting effectively down-ranks common terms (the, is, and) and amplifies discriminative spam indicators (viagra, click here, urgent)
- **Distributed approach scales:** Broadcast variables and partition-level caching enable efficient processing across clusters

## Project Structure

| File | Purpose |
|------|---------|
| `src/main/scala/SpamClassifier.scala` | Main entry point; orchestrates pipeline: load > preprocess > vectorize > classify > evaluate |
| `src/main/scala/Preprocessing.scala` | CSV parsing, text cleaning, tokenization, 80/20 train/test split |
| `src/main/scala/BagOfWords.scala` | Bag of Words feature extraction (word count vectors) |
| `src/main/scala/TFIDF.scala` | TF-IDF vectorization with distributed IDF computation |
| `src/main/scala/KNN.scala` | k-NN classifier with cosine similarity distance metric |
| `src/main/scala/Evaluation.scala` | Metrics computation: accuracy, precision, recall, F1, confusion matrix |
| `build.sbt` | Scala build configuration; declares Spark dependency |
| `data/spam_Emails_data.csv` | Dataset: 8,000 labeled emails |

## Results

| Feature Extraction | Model | Accuracy | Precision | Recall | F1-Score |
|-------------------|-------|----------|-----------|--------|----------|
| **TF-IDF** | **k-NN (k=5)** | **90.74%** | **90.26%** | **89.27%** | **89.76%** |
| Bag of Words | k-NN (k=5) | 70.73% | 66.09% | 73.18% | 69.45% |

**Confusion Matrix (TF-IDF):**
- True Positives: 649 | False Positives: 70
- True Negatives: 802 | False Negatives: 78

## Implementation

**Text Preprocessing:**
- Lowercase conversion and punctuation removal
- Tokenization on whitespace
- Email-specific stopword filtering (HTML tags, URLs, web artifacts)

**Feature Extraction:**
- **Bag of Words:** Raw word frequency counts per document
- **TF-IDF:** TF normalized by max count; IDF = log(N / document_frequency)

**Classification:**
- k-NN with cosine similarity: distance = dot product / (magnitude₁ × magnitude₂)
- Majority voting among k=5 nearest neighbors

**Distributed Computing:**
- Spark RDD operations for parallel data loading and feature extraction
- Broadcast variables to share training data across executors
- `mapPartitions` for memory-efficient batch prediction

## Requirements

- JDK 8 or higher
- Scala 2.11.8
- sbt (Scala Build Tool)
- Apache Spark 2.4.8

## Usage

```bash
# Build
sbt package

# Run with defaults (k=5, full dataset)
sbt run

# Run with custom parameters
sbt "run data/spam_Emails_data.csv 7"
```

**Parameters:**
- `dataPath`: Path to email CSV (default: `data/spam_Emails_data.csv`)
- `k`: Number of neighbors (default: `5`)
- `sampleSize`: Optional subsample size; `0` = full dataset (default: `0`)
- `outputPath`: Output file for results (default: `output.txt`)

## Notes

- **Vocabulary:** 76,443 unique terms after preprocessing
- **Runtime:** ~86 seconds on local machine
- **Class distribution:** 47% spam, 53% ham

## License

Educational project created for CSC369 (Big Data Fundamentals).
