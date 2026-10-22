# IMDB Sentiment Analysis

A sentiment analysis project developed during **Phase II of my AI internship at Reconess**. It classifies movie reviews as **positive** or **negative** using traditional machine learning and an LSTM neural network.

**Best result:** Logistic Regression with TF-IDF reached **90.05% accuracy** and **90.17% F1-score** on the held-out test set.

## Project Overview

This project explores the full NLP workflow:

1. **Explore the data:** Check class balance, missing values, duplicates, and review lengths.
2. **Clean the text:** Remove HTML tags, lowercase text, expand common negative contractions, and replace punctuation with spaces.
3. **Extract features:** Compare Bag of Words (BoW) and TF-IDF.
4. **Train models:** Test Logistic Regression and Linear SVM, then tune the best pipeline using GridSearchCV.
5. **Train an LSTM:** Use Keras TextVectorization, Embedding, and LSTM layers.
6. **Evaluate:** Compare both final models on the same held-out test set.

The final preprocessing preserves negation words such as `not`. Lemmatization and stopword removal were **not** used in the final cleaning pipeline.

## Dataset

The [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) contains 25,000 positive and 25,000 negative English-language reviews.

| Dataset | Number of reviews |
|---|---:|
| Original dataset | 50,000 |
| After duplicate removal and cleaning | 49,576 |
| Training set (80%) | 39,660 |
| Test set (20%) | 9,916 |

The test set was kept separate from model selection and hyperparameter tuning. Training data was split again into training and validation subsets for experiments.

## Models and Results

Four traditional model combinations were tested: Logistic Regression and Linear SVM, each with BoW and TF-IDF. **Logistic Regression + TF-IDF** performed best and was tuned with 3-fold GridSearchCV. An LSTM model was trained separately for comparison.

**Final test results (9,916 reviews):**

| Metric | Logistic Regression + TF-IDF | LSTM |
|---|---:|---:|
| Accuracy | **90.05%** | 88.46% |
| Precision | **89.38%** | 88.03% |
| Recall | **90.98%** | 89.13% |
| F1-score | **90.17%** | 88.58% |

Logistic Regression performed better in these experiments, showing that a simpler model can work very well for text classification. **This is not a perfectly controlled comparison:** the final Logistic Regression used 39,660 training reviews, while the LSTM used 31,728, and only Logistic Regression received the documented grid search.

## Project Structure

```text
imdb-sentiment-analysis/
├── notebooks/
│   ├── data_exploration_01.ipynb
│   ├── text_preprocessing_02.ipynb
│   ├── text_vectorization_03.ipynb
│   ├── model_training_04.ipynb
│   ├── lstm_training_05.ipynb
│   └── final_evaluation_06.ipynb
├── src/
│   └── migrate_lstm_vocabulary.py
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

The local project also uses `data/` for CSV files and `models/` for saved models. **These files are excluded from GitHub** because of `.gitignore` and must be obtained or shared separately.

## Installation

The project was developed and verified on **Windows with Python 3.11.9**. Open PowerShell in the project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` lists the main project dependencies; `requirements-lock.txt` records the tested environment's installed versions. A fresh installation and other operating systems have not been verified.

Download the dataset from the Kaggle link above and place the original file at `data/IMDB Dataset.csv`. The notebooks currently contain absolute Windows paths beginning with `D:/imdb-sentiment-analysis/`; update those paths if your project is stored elsewhere.

## Notebook Order

Run the notebooks in this order:

1. `data_exploration_01.ipynb` — Understand the dataset.
2. `text_preprocessing_02.ipynb` — Clean reviews and create `train.csv` and `test.csv`.
3. `text_vectorization_03.ipynb` — Explore BoW and TF-IDF.
4. `model_training_04.ipynb` — Train, compare, and tune traditional models.
5. `lstm_training_05.ipynb` — Train and evaluate the LSTM.
6. `final_evaluation_06.ipynb` — Evaluate both final models on the test set.

Training and grid search can take time. Existing notebook outputs can be reviewed without rerunning those steps.

## Using the Saved Models

If you have the saved model files, load the final Logistic Regression pipeline like this:

```python
import joblib

model = joblib.load("models/lr_tfidf_final.joblib")
prediction = model.predict(["this movie was wonderful"])
print(prediction[0])  # positive or negative
```

For LSTM inference, use **`models/lstm_model_portable.keras`**, which avoids an encoding issue found in the original Keras model:

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/lstm_model_portable.keras")
reviews = tf.constant([["this movie was wonderful"]], dtype=tf.string)
probability = model.predict(reviews, verbose=0)[0, 0]
print("positive" if probability > 0.5 else "negative")
```

**Important:** Both saved models expect text cleaned with the preprocessing steps from notebook 02. The examples above use already-simple text; real raw reviews (including HTML and contractions) should be preprocessed first. The Logistic Regression artifact contains its fitted TF-IDF vectorizer, and the LSTM contains its TextVectorization layer, so neither needs to be fitted again for inference.

## Key Takeaways

- TF-IDF worked better than BoW in the recorded traditional-model experiments.
- Tuned Logistic Regression achieved the strongest final test results.
- The LSTM learned sentiment patterns but showed some early overfitting.
- Future improvements include more consistent model comparisons, reusable preprocessing code, and broader environment testing.

This project demonstrates a complete introductory NLP pipeline, from text cleaning and feature extraction to model training and final evaluation.
