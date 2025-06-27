# Summer of Code (SOC) 2025 – Learning Report

## Overview

During the Summer of Code (SOC) 2025, I undertook a deep dive into four major domains:
1. Python Programming
2. Machine Learning
3. Natural Language Processing (NLP) using the YouTube course by Codebasics
4. NLP fundamentals via Coursera's "Natural Language Processing with Classification and Vector Spaces" by DeepLearning.AI

This report documents, in-depth, all that I have learned through video lectures, documentation, assignments, hands-on coding, and self-initiated projects during the course of SOC.

---

## 1. Python Programming

### Purpose and Role in My Learning
Python served as the foundational tool for all subsequent topics I studied. A solid grasp of Python was essential for implementing algorithms, manipulating data, and experimenting with models.

### Concepts Learned

#### Basic Syntax and Operations
- Variables, expressions, and data types (integers, floats, strings, booleans)
- Arithmetic, logical, and comparison operations
- Type conversion and input/output functions

#### Control Structures
- `if`, `elif`, and `else` statements for decision making
- `while` and `for` loops with `range()`
- `break`, `continue`, and `pass` keywords

#### Functions and Modules
- Defining and calling functions with and without return values
- Arguments: positional, keyword, default, and variable-length arguments
- `import`, `from`, and module structure

#### Data Structures
- **Lists**: creation, indexing, slicing, methods like `append`, `insert`, `pop`, and list comprehensions
- **Tuples**: immutability, tuple unpacking
- **Dictionaries**: key-value pairs, methods like `get`, `keys`, `values`, `items`
- **Sets**: uniqueness, operations like union, intersection, difference

#### File Handling
- Reading and writing text and CSV files using `open()`, `.read()`, `.write()`
- Using the `with` statement for safe file handling

#### Exception Handling
- Try-except blocks, `finally` clause, and raising custom errors

### Libraries Learned

#### NumPy
- Arrays: creation, indexing, slicing, broadcasting
- Vectorized operations and mathematical functions
- Array reshaping and dimension manipulation

#### Pandas
- Series and DataFrames
- Importing datasets (`read_csv`, `read_excel`)
- Data inspection (`head`, `info`, `describe`)
- Filtering, sorting, and grouping data
- Handling missing values (`isnull`, `fillna`, `dropna`)

#### Visualization: Matplotlib and Seaborn
- Line, bar, scatter, histogram, and box plots
- Customizing axes, labels, legends
- Seaborn aesthetics and statistical plots like `pairplot`, `heatmap`

---

## 2. Machine Learning

### Overview
After building a foundation in Python, I moved on to classical machine learning using the `scikit-learn` library. This section involved both theory and implementation, focusing on predictive models.

### Concepts and Algorithms

#### Supervised Learning

**Linear Regression**
- Understanding the line of best fit
- Cost function and gradient descent
- Using `LinearRegression()` in `sklearn`

**Logistic Regression**
- Classification vs regression
- Sigmoid function and decision boundary
- Binary and multiclass classification

**Decision Trees and Random Forests**
- Tree structure, splitting criteria (Gini, Entropy)
- Overfitting and pruning
- Ensemble learning and feature bagging with random forests

**K-Nearest Neighbors (KNN)**
- Distance metrics (Euclidean)
- Choosing optimal k-value
- Pros and cons: simplicity vs computational cost

**Naive Bayes**
- Bayes theorem and independence assumption
- Multinomial vs Gaussian models
- Good for text classification

**Support Vector Machines (SVM)**
- Margin maximization and support vectors
- Linear vs kernel SVM (RBF)
- Using `SVC()` in `sklearn`

### Model Evaluation Techniques
- **Confusion Matrix**: TP, TN, FP, FN
- **Accuracy, Precision, Recall, F1-score**
- **ROC Curve and AUC**: Visual evaluation of classification
- **K-Fold Cross Validation**: Reducing variance and overfitting

### Feature Engineering
- **Handling Missing Values**: mean/median imputation, dropping
- **Encoding Categorical Variables**: One-hot encoding, label encoding
- **Scaling Features**: Min-Max Scaler, Standard Scaler
- **Polynomial Features**: Capturing non-linearity

### Project Exercises
- Predicting house prices using linear regression
- Classifying iris species with KNN
- Detecting spam messages with Naive Bayes

---

## 3. NLP YouTube Tutorial Series by Codebasics

### Course Link
https://www.youtube.com/playlist?list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12

### Text Preprocessing

#### Techniques
- **Tokenization**: Splitting sentences into words
- **Stopword Removal**: Removing common but insignificant words ("the", "is")
- **Stemming**: Reducing words to base form ("playing" → "play") using `PorterStemmer`
- **Lemmatization**: More accurate root word extraction using `WordNetLemmatizer`

### Feature Extraction

**Bag of Words (BoW)**
- Creating a vocabulary of known words
- Representing documents as word occurrence vectors

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Enhancing BoW by down-weighting frequent/common words
- More meaningful representation for classification

### Classification Models
- **Spam Detection Project**: Used BoW + Multinomial Naive Bayes
- **Sentiment Analysis**: Used logistic regression on TF-IDF vectors

### Learnings
- Text preprocessing has a large impact on model performance
- Bag of Words is intuitive but sparse; TF-IDF provides more nuance
- Logistic regression and Naive Bayes perform well on text datasets

---

## 4. Coursera: NLP with Classification and Vector Spaces

### Course Link
https://www.coursera.org/learn/classification-vector-spaces-in-nlp

### Week-by-Week Learning

#### Week 1: Naive Bayes Text Classification
- Probabilistic modeling using word counts
- Log likelihood and additive smoothing
- Manual implementation of Naive Bayes classifier

#### Week 2: Vector Semantics
- **Word Co-occurrence Matrix**: Representing word meaning through context
- **TF-IDF Weighting**: Transforming word counts to reflect significance
- **Cosine Similarity**: Measuring text similarity using angle between vectors

#### Week 3: Classification with k-NN
- Non-parametric model using distance in vector space
- Using TF-IDF vectors with cosine similarity in nearest neighbor search
- Limitations with large datasets and sparse vectors

### Practical Assignments
- Implemented text classifiers from scratch in Python
- Created and normalized document-term matrices
- Visualized vector relationships and document similarities

### Reflections
- The mathematical grounding added depth to my understanding of text representations
- Comparing multiple techniques showed how accuracy depends on preprocessing and vectorization
- Realized the importance of performance metrics like cosine similarity in high-dimensional spaces

---

## Summary and Key Takeaways

- Python is a versatile tool that underpins both machine learning and NLP workflows.
- Supervised ML models have varied strengths and choosing the right model requires experimentation.
- Preprocessing and feature representation are critical in NLP tasks; often more than model selection.
- Understanding vector semantics and probability is key to mastering foundational NLP.

---

## References

- **Codebasics YouTube Channel**: For clear and hands-on NLP explanations
- **DeepLearning.AI & Coursera**: For academic rigor and mathematical foundation

---

