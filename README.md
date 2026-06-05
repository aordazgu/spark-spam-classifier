# Spark Spam Classifier

A distributed machine learning system for email spam classification using Apache Spark and k-NN in Scala. Compares Bag of Words and TF-IDF feature representations on 8,000 real emails, demonstrating how feature engineering impacts classification performance.

**Key Achievement**: TF-IDF-based classifier achieves **90.74% accuracy** with 90.26% precision and 89.27% recall—significantly outperforming Bag of Words (70.73%).

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **TF-IDF + k-NN** | **90.74%** | **90.26%** | **89.27%** | **89.76%** |
| Bag of Words + k-NN | 70.73% | 66.09% | 73.18% | 69.45% |

The TF-IDF approach significantly outperforms simple word frequency counting, demonstrating the value of inverse document frequency weighting for spam detection.

## 🏗️ Project Structure

```
src/main/scala/
├── SpamClassifier.scala    # Pipeline orchestration: load → vectorize → classify → evaluate
├── Preprocessing.scala      # CSV parsing, text cleaning, 80/20 train/test split
├── BagOfWords.scala         # Word count feature extraction
├── TFIDF.scala              # TF-IDF vectorization with distributed IDF computation
├── KNN.scala                # k-NN classifier with cosine similarity
└── Evaluation.scala         # Metrics: accuracy, precision, recall, F1, confusion matrix
```

**Design**: Modular, single-responsibility components with clean RDD transformations. Each module encapsulates one stage of the ML pipeline.

## 🎯 Architecture

- **Preprocessing**: Text cleaning, tokenization, and email-specific stopword removal
- **Feature Extraction**: 
  - Bag of Words (baseline word counts)
  - TF-IDF (term frequency × inverse document frequency weighting)
- **Classification**: k-NN with cosine similarity; broadcast training data for efficiency
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices

## 🚀 Getting Started

### Prerequisites
- JDK 8 or higher
- Scala 2.11.8
- sbt (Scala build tool)

### Build
```bash
sbt package
```

### Run

**Default (uses all 8,000 emails with k=5):**
```bash
sbt run
```

**Custom parameters:**
```bash
sbt "run <dataPath> <k> [sampleSize] [outputPath]"
```

**Example:**
```bash
sbt "run data/spam_Emails_data.csv 7 4000"
```

**Parameter Reference:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataPath` | `data/spam_Emails_data.csv` | Path to CSV file with labeled emails |
| `k` | `5` | Number of neighbors for k-NN classification |
| `sampleSize` | `0` (full dataset) | Optional: subsample to N emails for quick testing |
| `outputPath` | `output.txt` | File to write accuracy, precision, recall, F1 metrics |

## 📊 Dataset

- **Size**: 8,000 emails (3,779 spam, 4,221 ham)
- **Split**: 80% training, 20% testing
- **Vocabulary**: 76,443 unique terms after preprocessing

## 🔧 Implementation Details

### Text Preprocessing
Emails are cleaned with:
- Lowercase conversion
- Removal of punctuation and special characters
- Stopword filtering (HTML tags, URLs, common web artifacts)
- Tokenization on whitespace

### TF-IDF Scoring
TF (term frequency) is normalized by maximum term count per document:
```
TF(term, doc) = count(term, doc) / max_count(doc)
```

IDF (inverse document frequency) is computed as:
```
IDF(term) = log(total_documents / documents_containing_term)
```

### Cosine Similarity
Distance is measured as normalized dot product of feature vectors, ensuring scale-invariant comparison.

### Distributed Processing
The system leverages Apache Spark for:
- Parallel data loading and parsing
- Distributed feature extraction across partitions
- Efficient broadcasting of training data to avoid duplication
- Distributed prediction with `mapPartitions` for memory efficiency

## ⚡ Technical Highlights

**Distributed Computing**
- Spark RDD operations for parallel data loading, feature extraction, and prediction
- Broadcast variables to efficiently share training data across partitions (avoids duplication)
- `mapPartitions` for memory-efficient batch processing of test emails

**Feature Engineering**
- Custom stopword list targeting email-specific artifacts (HTML tags, URLs, web headers)
- TF normalization by max count per document (L∞ norm)
- IDF computed via distributed MapReduce: count documents per term, then apply log weighting

**k-NN Optimization**
- Cosine similarity for scale-invariant, sparse-vector-friendly distance metric
- Majority voting among top-k neighbors
- Partitioned computation avoids redundant distance calculations

## 💡 Why TF-IDF Significantly Outperforms Bag of Words

Bag of Words treats all words equally, assigning high weight to frequently appearing terms like "the", "is", "and"—words present in both spam and ham. TF-IDF solves this by downweighting common terms and amplifying rarer, more discriminative ones. In spam detection, this means words like "viagra", "click here", "urgent", or "limited offer" get high IDF values because they appear disproportionately in spam. The 20-point accuracy gap (90.74% vs 70.73%) demonstrates the practical impact of this insight.

## 📈 Performance

- **Runtime**: ~86 seconds on local machine with 8,000 emails
- **Memory**: Efficient partitioning and broadcasting reduce per-executor memory overhead

## 🚀 Potential Improvements

- **Hyperparameter Tuning**: Grid search over k ∈ {1, 3, 5, 7, 9} to optimize neighbor count
- **Alternative Classifiers**: Naive Bayes or SVM via Spark MLlib for comparison
- **Dimensionality Reduction**: PCA or feature selection to reduce vocabulary from 76k terms
- **Error Analysis**: Examine misclassified emails to identify challenging patterns
- **Cross-Validation**: Stratified k-fold for more robust evaluation
- **Real-Time Processing**: Spark Streaming for online email classification

## 🎓 Technical Skills Demonstrated

- **Big Data**: Distributed processing with Apache Spark and RDDs
- **ML**: Feature engineering, vectorization, k-NN classification, evaluation metrics
- **NLP**: Text preprocessing, tokenization, stopword removal
- **Engineering**: Modular design, MapReduce patterns, memory-efficient broadcasting
- **Programming**: Scala functional paradigms, case classes, option handling

## 📝 Notes

- **Dataset**: 8,000 emails (47% spam, 53% ham) from real-world sources
- **Train/Test**: 80/20 split with stratified sampling to preserve class distribution
- **Vocabulary**: 76,443 unique terms after preprocessing
- **Vocabulary Size**: Email-specific preprocessing with HTML, URL, and web artifact filtering significantly reduces noise

## 📝 License

Educational project created for CSC369 (Big Data Fundamentals).

## 👤 Author

Arturo Ordaz-Gutierrez
