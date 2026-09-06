# 💬 Sentify — 3-Class Social Media Sentiment Analysis

Sentify is a **Natural Language Processing (NLP) and Machine Learning project** that analyzes social media posts and classifies them into three sentiment categories:

* 🟢 Positive
* ⚪ Neutral
* 🔴 Negative

The project uses **TF-IDF** for text feature extraction and compares **Logistic Regression** and **Linear SVM** for sentiment classification.

---

## 📌 Features

* Social media text preprocessing
* Missing-value handling
* Duplicate removal
* Irrelevant sentiment removal
* Text cleaning
* TF-IDF feature extraction
* Multi-class sentiment classification
* Logistic Regression
* Linear SVM
* Accuracy, Precision, Recall and F1-Score
* Classification Report
* Confusion Matrix
* Word-frequency analysis
* Exploratory Data Analysis

---

## 📊 Dataset

This project uses the **Twitter Entity Sentiment Analysis** dataset.

The dataset contains social media posts associated with different entities/topics and their corresponding sentiment labels.

### Dataset Files

```text
twitter_training.csv
twitter_validation.csv
```

### Dataset Columns

| Column      | Description                           |
| ----------- | ------------------------------------- |
| `id`        | Unique identifier                     |
| `entity`    | Entity/topic associated with the post |
| `sentiment` | Sentiment label                       |
| `text`      | Social media post                     |

The original dataset contains four sentiment categories:

```text
Positive
Negative
Neutral
Irrelevant
```

For this project, the `Irrelevant` class is removed and the problem is treated as a **3-class classification task**:

```text
Positive
Neutral
Negative
```

---

## 📈 Dataset Statistics

### Training Dataset

* Original records: **74,682**
* Missing text values: **686**
* Duplicate tweets identified: **3,824**
* Final sentiment classes:

  * Positive
  * Neutral
  * Negative

### Validation Dataset

* Records: **1,000**

---

## 🔄 Project Workflow

```text
                ┌──────────────────────┐
                │   Raw Twitter Data   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Data Cleaning        │
                │ • Missing Values     │
                │ • Duplicates         │
                │ • Irrelevant Class   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Text Preprocessing   │
                │ • Lowercase          │
                │ • URL Removal        │
                │ • Mention Removal    │
                │ • Punctuation       │
                │ • Number Removal     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ TF-IDF Vectorization │
                └──────────┬───────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   Machine Learning       │
              │                          │
              │ • Logistic Regression    │
              │ • Linear SVM             │
              └────────────┬─────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Model Evaluation      │
                │ • Accuracy            │
                │ • Precision           │
                │ • Recall              │
                │ • F1-Score            │
                │ • Confusion Matrix    │
                └──────────────────────┘
```

---

## 🧹 Data Preprocessing

The social media text contains noise such as URLs, mentions, hashtags, punctuation and numbers.

The following preprocessing operations are performed:

### 1. Remove Missing Values

Rows with missing `text` or `sentiment` values are removed.

### 2. Normalize Sentiment Labels

Sentiment labels are converted to lowercase and whitespace is removed.

```python
train_df["sentiment"] = (
    train_df["sentiment"]
    .astype(str)
    .str.lower()
    .str.strip()
)
```

### 3. Remove Irrelevant Sentiment

Only the following classes are retained:

```python
valid_labels = ["positive", "negative", "neutral"]
```

### 4. Remove Duplicate Records

Duplicate tweets are identified using the text and sentiment columns and then removed.

### 5. Clean Text

The text-cleaning pipeline performs:

* Lowercasing
* URL removal
* Mention removal
* Hashtag symbol removal
* Punctuation removal
* Number removal
* Extra whitespace removal

Example:

```text
Before:
"I LOVE this game!!! Check https://example.com @user #Gaming"

After:
"i love this game gaming"
```

---

## 🔤 TF-IDF Feature Extraction

Machine-learning algorithms cannot directly process raw text.

Therefore, the cleaned tweets are converted into numerical feature vectors using:

### TF-IDF

**TF-IDF = Term Frequency × Inverse Document Frequency**

TF-IDF gives greater importance to words that are useful for distinguishing documents while reducing the importance of extremely common words.

The resulting TF-IDF vectors are used as input to the machine-learning models.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as one of the baseline classifiers.

It is suitable for text classification because TF-IDF produces high-dimensional and sparse feature representations.

### 2. Linear SVM

A **Linear Support Vector Machine** is also trained for sentiment classification.

Linear SVM is particularly useful for high-dimensional text classification problems.

---

## 📏 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many predicted samples of a class are actually correct.

### Recall

Measures how many actual samples of a class are correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Classification Report

A detailed classification report is generated for:

```text
Negative
Neutral
Positive
```

### Confusion Matrix

A confusion matrix is used to analyze how predictions are distributed among the three sentiment classes.

---

## 📊 Exploratory Data Analysis

The project performs several exploratory analyses.

### Sentiment Distribution

A bar chart is used to visualize the distribution of Positive, Neutral and Negative tweets.

### Tweet Length Analysis

The project calculates the length of cleaned tweets.

| Statistic          |  Value |
| ------------------ | -----: |
| Mean               | 112.37 |
| Standard Deviation |  78.58 |
| Minimum            |      1 |
| Median             |     95 |
| Maximum            |    957 |

### Word Frequency Analysis

The project identifies the most frequently occurring words in the dataset.

A frequency analysis and visualization of the top 20 words are included in the notebook.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Linear SVM

### NLP

* TF-IDF
* Regular Expressions
* Text Preprocessing

### Visualization

* Matplotlib
* WordCloud

### Model Saving

* Joblib

---

## 📁 Project Structure

```text
Sentify/
│
├── README.md
│
├── analysis.ipynb
│
├── twitter_training.csv
├── twitter_validation.csv
│
├── app.py
│
├── requirements.txt
│
└── models/
    ├── sentiment_model.pkl
    └── tfidf_vectorizer.pkl
```

> Update the filenames inside the `models/` folder according to the actual files in your repository.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project

```bash
cd Sentify
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
analysis.ipynb
```

Run the notebook cells sequentially to reproduce the complete data-processing, training and evaluation pipeline.

---

## 🌐 Running the Streamlit Application

If `app.py` is included in the repository:

```bash
streamlit run app.py
```

The application can then be accessed through the local Streamlit URL displayed in the terminal.

---

## 📌 Example Prediction

Input:

```text
I absolutely love this game!
```

Output:

```text
Sentiment: Positive
```

Input:

```text
This game is terrible and disappointing.
```

Output:

```text
Sentiment: Negative
```

Input:

```text
The game was released yesterday.
```

Output:

```text
Sentiment: Neutral
```

---

## 🎯 Applications

Sentify can be used for:

* 📱 Social media monitoring
* 💬 Customer feedback analysis
* 🎮 Gaming community analysis
* 🛍️ Product review analysis
* 📢 Brand sentiment monitoring
* 📊 Public opinion analysis
* 💻 Online community analytics

---


## 📚 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Natural Language Processing
* Exploratory Data Analysis
* Data Cleaning
* Text Preprocessing
* Feature Engineering
* TF-IDF
* Multi-class Classification
* Logistic Regression
* Linear SVM
* Model Evaluation
* Confusion Matrix
* Data Visualization
* Machine Learning Pipeline

---

## 👩‍💻 Author

### Sayani Jana

**M.Tech — Computer Science / Data Science**


## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
