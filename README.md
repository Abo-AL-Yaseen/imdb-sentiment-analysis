# IMDB Sentiment Analysis

An end-to-end movie review classification project completed during **Phase II of the Reconess AI internship**. The project uses Python, Pandas, Scikit-learn, TensorFlow, and Keras to classify reviews as **positive** or **negative**.

The workflow covers data exploration, text cleaning, Bag of Words (BoW), TF-IDF, traditional classifiers, hyperparameter tuning, an LSTM network, and final evaluation. **Tuned Logistic Regression with TF-IDF achieved the best recorded final results: 90.05% accuracy and 90.17% F1-score.**

## 1. Project Objectives

- Prepare and inspect a real sentiment analysis dataset while preserving useful negation cues.
- Compare Logistic Regression and Linear SVM with BoW and TF-IDF features.
- Tune the strongest baseline and compare it with an LSTM on a reserved final test set.

## 2. Dataset Description and Source

Source: [IMDB Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

The source file, `data/IMDB Dataset.csv`, contains 50,000 English movie reviews: 25,000 positive and 25,000 negative. Its columns are `review` (text) and `sentiment` (label).

After removing exact duplicates and duplicates introduced by cleaning, the project uses **49,576 unique cleaned reviews**. The exported CSVs have the same two column names, but their `review` column contains **preprocessed text**.

| Dataset | Reviews | Positive | Negative |
|---|---:|---:|---:|
| Raw dataset | 50,000 | 25,000 | 25,000 |
| After exact duplicate removal | 49,582 | 24,884 | 24,698 |
| After cleaned-text duplicate removal | 49,576 | 24,881 | 24,695 |
| `data/train.csv` | 39,660 | 19,904 | 19,756 |
| `data/test.csv` | 9,916 | 4,977 | 4,939 |

The final 80/20 split uses `random_state=42` and `stratify=y`. Training and test reviews have no exact text overlap. The final test data is reserved for final evaluation; model selection uses a separate validation split of `train.csv`.

## 3. Project Structure

```text
imdb-sentiment-analysis/
├── data/
│   ├── IMDB Dataset.csv
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── lr_tfidf_tuned.joblib
│   ├── lr_tfidf_final.joblib
│   ├── lstm_model.keras
│   └── lstm_model_portable.keras
├── notebooks/
│   ├── data_exploration_01.ipynb
│   ├── text_preprocessing_02.ipynb
│   ├── text_vectorization_03.ipynb
│   ├── model_training_04.ipynb
│   ├── lstm_training_05.ipynb
│   └── final_evaluation_06.ipynb
├── src/
│   └── migrate_lstm_vocabulary.py
├── .gitignore
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

The LSTM notebook's actual filename is **`lstm_training_05.ipynb`**. The directory above describes the inspected local project. Datasets and model artifacts are excluded by `.gitignore`, so they must be supplied separately when sharing the project.

## 4. Technologies and Libraries

| Tool | Role |
|---|---|
| Python 3.11.9 | Runtime used by the saved notebook kernels and verification environment |
| Pandas / NumPy | Data inspection, splitting inputs, and array operations |
| Python `re` | HTML removal and text normalization |
| Scikit-learn | Vectorizers, pipelines, classifiers, GridSearchCV, and metrics |
| TensorFlow / Keras | TextVectorization, LSTM training, and model serialization |
| Matplotlib | Review-length histogram, confusion matrix, and learning curves |
| NLTK | Preprocessing exploration; lemmatization is excluded from the final pipeline |
| Joblib | Saving and loading Scikit-learn pipelines |
| VS Code / Jupyter notebooks / ipykernel | Interactive development and execution |

`requirements.txt` pins the project dependencies to versions from the working environment, including explicit Keras and Joblib dependencies. `requirements-lock.txt` records the installed dependency versions, excluding packaging tools. spaCy remains listed from the original environment but is not used by the current notebooks.

## 5. Exploratory Data Analysis

Notebook 01 records no missing values, **418 duplicate rows/reviews**, and no identical reviews with conflicting labels. Removing those duplicates leaves 49,582 reviews.

Word counts are measured with whitespace splitting before text preprocessing:

| Statistic | Words per review |
|---|---:|
| Mean | 231.35 |
| Median | 173 |
| 25th / 75th percentile | 126 / 281 |
| Minimum / maximum | 4 / 2,470 |

The histogram shows a long right tail: most reviews are relatively short, while a smaller number are much longer. Average lengths are similar for negative and positive reviews: approximately 229.60 and 233.09 words, respectively.

## 6. Text Preprocessing

Notebook 02 applies the following steps:

1. Remove exact duplicate rows.
2. Replace basic HTML tags with spaces using `<.*?>`.
3. Convert text to lowercase.
4. Expand 18 selected negative contractions, for example `didn't` → `did not` and `can't` → `can not`.
5. Replace punctuation with **spaces**, using `re.sub(r"[^\w\s]", " ", text)`.
6. Split on whitespace and join with single spaces.
7. Remove duplicate preprocessed reviews, then create the stratified train/test split.

This preserves negation words such as `not` and `never`. **No stopword removal, stemming, or lemmatization is applied by `preprocess_text()`.** NLTK resources are downloaded in an exploratory cell, but those resources are not needed for the final cleaning function. The original text remains available in memory alongside `preprocessed_review`; only cleaned text and labels are exported.

The saved outputs and CSV verification show **6 additional duplicates** after cleaning, with no missing values, empty reviews, or conflicting-label groups. Notebook 02's Markdown now matches these counts and the implemented punctuation replacement. Its code, calculations, execution counts, and experiment outputs were preserved.

## 7. Feature Extraction: BoW and TF-IDF

- **BoW:** `CountVectorizer(max_features=5000)` represents reviews using word counts.
- **TF-IDF:** `TfidfVectorizer(max_features=5000)` weights words by their frequency in a review and their rarity across training reviews.
- Both use word unigrams and their default tokenization, without a stopword list.

Notebook 03 demonstrates both representations: vectorizers fit only on the 39,660 training reviews and transform the 9,916 test reviews. Each resulting matrix has 5,000 columns. These exploratory vectorizers are not saved or reused by notebook 04.

Notebook 04 builds fresh `Pipeline` objects after making its training/validation split. This keeps vocabulary and IDF learning inside the training subset, including inside each GridSearchCV fold. Notebook 03 transforms test text for demonstration but does not evaluate classifiers or select settings on it.

## 8. Traditional ML Models

Notebook 04 splits `train.csv` into **31,728 training reviews** and **7,932 validation reviews**, using a stratified 80/20 split with `random_state=42`.

It compares Logistic Regression (`max_iter=1000`, `random_state=42`) and `LinearSVC(random_state=42)` using the same validation reviews. Each baseline vectorizer has a 5,000-feature limit.

**Recorded validation results** — positive sentiment is the positive class:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression + TF-IDF | **88.44%** | **88.28%** | **88.75%** | **88.51%** |
| Linear SVM + TF-IDF | 87.67% | 87.70% | 87.74% | 87.72% |
| Logistic Regression + BoW | 86.03% | 86.19% | 85.93% | 86.06% |
| Linear SVM + BoW | 84.27% | 84.62% | 83.90% | 84.26% |

The saved confusion matrix for baseline Logistic Regression + TF-IDF has TN=3,482, FP=469, FN=448, and TP=3,533. TF-IDF outperforms BoW for both classifiers in these recorded experiments.

## 9. Hyperparameter Tuning

`GridSearchCV` tunes the Logistic Regression + TF-IDF pipeline on the **31,728-review training subset**, using three-fold cross-validation and F1 scoring with `pos_label="positive"`.

| Parameter | Values searched | Recorded best value |
|---|---|---:|
| `tfidf__max_features` | 5,000; 10,000; 20,000; 500,000; 1,000,000 | 500,000 |
| `classifier__C` | 0.1; 1; 10; 100; 200 | 10 |

There are 25 combinations and 75 cross-validation fits, followed by refitting the best pipeline. The recorded best mean CV F1-score is **0.8952395233** (89.52%).

| Tuned model on validation set | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression + TF-IDF | 88.98% | 88.89% | 89.20% | 89.04% |

The saved `lr_tfidf_tuned.joblib` reproduces these validation metrics. Its actual vocabulary has **84,660 features**: `max_features=500000` is an upper limit, not the fitted vocabulary size.

## 10. LSTM Deep Learning Model

Notebook 05 uses the same 31,728/7,932 training/validation split as notebook 04. Labels are encoded as `positive=1` and `negative=0`.

```text
String input, shape (batch, 1)
  → TextVectorization (10,000 tokens, integer sequences of length 300)
  → Embedding (10,000 × 32, mask_zero=True)
  → LSTM (32 units)
  → Dense (1 unit, sigmoid)
```

The vectorizer adapts only on training text. It uses lowercase/punctuation standardization and whitespace splitting, pads shorter sequences, and truncates longer ones to 300 tokens. Its vocabulary includes padding and out-of-vocabulary entries. The network has **328,353 trainable parameters**.

Training uses Adam, binary cross-entropy, accuracy monitoring, batches of 64, and **3 epochs**. The training dataset is shuffled with a 10,000-example buffer and seed 42; both datasets use prefetching. Model initialization has no explicit global random seed. No dropout or early stopping is configured.

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 79.61% | 0.4290 | 82.35% | 0.4151 |
| 2 | 89.94% | 0.2575 | 86.71% | 0.3086 |
| 3 | 92.37% | 0.2071 | 87.97% | 0.3233 |

The notebook includes accuracy and loss curves. The widening accuracy gap and increase in validation loss from epoch 2 to 3 suggest **early overfitting**, even though validation accuracy still improves. The final epoch is saved as `lstm_model.keras`.

## 11. Model Evaluation and Results

All metric calls use the correct **`(y_true, y_pred)`** argument order. Precision, recall, and F1 are binary metrics for positive sentiment, using `pos_label="positive"` for Logistic Regression and label `1` for the LSTM. LSTM predictions use **probability > 0.5** for positive sentiment.

The LSTM's recorded validation scores are 87.97% accuracy, 88.35% precision, 87.59% recall, and 87.97% F1. The original cell prints these scores to four decimal places.

In notebook 06, the tuned Logistic Regression pipeline is cloned and **retrained on all 39,660 available training reviews**, including the earlier validation subset. This refits both its TF-IDF vectorizer and classifier, producing **92,913 vocabulary features**. The LSTM is loaded from its saved artifact without retraining on the full training file.

Both models are then evaluated on the same **9,916 held-out test reviews**. The final test set is not used for fitting or hyperparameter selection in the implemented workflow.

## 12. Final Model Comparison

| Final test metric | Logistic Regression + TF-IDF | LSTM |
|---|---:|---:|
| Accuracy | **90.05%** | 88.46% |
| Precision | **89.38%** | 88.03% |
| Recall | **90.98%** | 89.13% |
| F1-score | **90.17%** | 88.58% |
| Training reviews used | 39,660 | 31,728 |

These values match notebook 06's executed outputs and were independently reproduced by loading the saved artifacts and predicting on `test.csv`, without retraining.

Logistic Regression performs best in the recorded experiments, with approximately **1.58 percentage points higher final accuracy**. However, the final models were trained on different numbers of reviews, and only Logistic Regression received the documented grid search. This is an imperfect comparison of model families. A more complex deep learning model does not necessarily outperform traditional machine learning on this task.

## 13. Installation and Environment Setup

The supported, inspected environment uses **Windows, Python 3.11.9**, TensorFlow 2.21.0, Keras 3.15.1, and Scikit-learn 1.9.1. All project requirements now use exact versions from this environment. For its complete application dependency snapshot, install `requirements-lock.txt`; for the smaller set of project dependencies, install `requirements.txt`.

From Windows PowerShell in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Reuse an existing `.venv` if it is already configured. These commands use the environment's Python directly, so activation is optional. To activate it in a PowerShell session that allows activation scripts, use `.\.venv\Scripts\Activate.ps1`.

The snapshot pins dependency versions, not distribution hashes or the Python runtime. It excludes `pip`, `setuptools`, and `wheel`. The working environment passes `pip check`, and every pin matches an installed distribution; no packages were installed or upgraded during cleanup. A fresh installation and other operating systems have not been tested.

In VS Code, install the Python and Jupyter extensions, open this project folder, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**, and choose the same environment as the notebook kernel.

**Original LSTM encoding issue:** In Keras 3.15.1, `IndexLookup.save_assets()` and `load_assets()` in `keras/src/layers/preprocessing/index_lookup.py` open `vocabulary.txt` without an explicit encoding. The original artifact was saved under Windows CP1252. Both vocabulary assets contain CP1252 bytes, for example `0xE9` in `cliché`, which are invalid UTF-8 at that position. The layer's `encoding="utf-8"` setting does not control these Python file reads. Consequently, the original `lstm_model.keras` loads under CP1252 but fails in Python UTF-8 mode. Check an interpreter's configuration with:

```powershell
.\.venv\Scripts\python.exe -c "import locale, sys; print(locale.getencoding(), sys.flags.utf8_mode)"
```

The original model's verified configuration prints `cp1252 0`. For inference, **use `models/lstm_model_portable.keras`**, which loads in both the tested CP1252 configuration and Python UTF-8 mode.

The portable copy stores the same ordered 10,000-entry vocabulary in the TextVectorization layer's supported JSON `vocabulary` field, with ASCII-escaped Unicode. It removes the two redundant vocabulary text assets, avoiding locale-dependent reads entirely. The weights/optimizer-state HDF5 member, metadata, architecture, hyperparameters, and token IDs remain unchanged. Simply transcoding the text assets to UTF-8 would still leave the default-encoding reader dependent on the receiving machine's locale.

The migration is reproducible without TensorFlow or retraining. Run this command from the project root **only when the destination does not already exist**; the script refuses to replace either the source or any existing destination:

```powershell
.\.venv\Scripts\python.exe -B src/migrate_lstm_vocabulary.py models/lstm_model.keras models/lstm_model_portable.keras
```

The original notebook loading cells still reference `lstm_model.keras`, preserving the historical experiments. Those cells retain the original CP1252 requirement; standalone inference below uses the portable copy. Actual Linux/macOS execution has not been tested.

## 14. How to Run the Project

The completed experiments can be reviewed through their saved notebook outputs. For a deliberate full reproduction, follow this order:

| Order | Notebook | Input and purpose | Files written |
|---:|---|---|---|
| 1 | `data_exploration_01.ipynb` | Inspect the raw dataset and review lengths | None |
| 2 | `text_preprocessing_02.ipynb` | Clean the raw data, deduplicate, and split | `data/train.csv`, `data/test.csv` |
| 3 | `text_vectorization_03.ipynb` | Demonstrate BoW and TF-IDF using the split CSVs | None |
| 4 | `model_training_04.ipynb` | Compare baselines and tune Logistic Regression | `models/lr_tfidf_tuned.joblib` |
| 5 | `lstm_training_05.ipynb` | Train, plot, and save the LSTM | `models/lstm_model.keras` |
| 6 | `final_evaluation_06.ipynb` | Refit final Logistic Regression and evaluate both models | `models/lr_tfidf_final.joblib` |

Download the raw CSV from the linked dataset source if it is absent and place it at `data/IMDB Dataset.csv`. Notebooks use absolute paths rooted at **`D:/imdb-sentiment-analysis/`**; running them elsewhere requires updating those paths. Notebook 03 is an independent demonstration, not an artifact dependency of notebook 04. Notebooks 04 and 05 independently consume notebook 02's training CSV; notebook 06 needs their saved models.

Full execution includes expensive grid search and training, and rerunning writing cells replaces the listed outputs. Preserve existing datasets and models before an intentional reproduction. For inference only, use the saved artifacts as shown next.

## 15. Saved Models and Their Usage

| Artifact | Contents and intended use |
|---|---|
| `models/lr_tfidf_tuned.joblib` | TF-IDF + tuned Logistic Regression fitted on 31,728 reviews; validation-stage artifact |
| `models/lr_tfidf_final.joblib` | Same settings refitted on all 39,660 training reviews; final Logistic Regression inference |
| `models/lstm_model.keras` | Final-epoch LSTM, embedded TextVectorization vocabulary, learned weights, and optimizer state |
| `models/lstm_model_portable.keras` | Same trained LSTM and vocabulary, packaged with inline JSON vocabulary for locale-independent loading |

**Both models expect the project's cleaned review text.** The Logistic Regression pipeline includes TF-IDF but excludes `preprocess_text()`. The LSTM includes token vectorization but excludes the external HTML removal and negation expansion. Therefore, normalize new raw reviews before inference. Reviews read from `train.csv` or `test.csv` are already cleaned.

The following helper reproduces notebook 02's current cleaning logic for standalone examples; it does not require NLTK resources:

```python
import re

def preprocess_text(text):
    text = re.sub(r"<.*?>", " ", text).lower()
    contractions = [
        "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
        "didn't", "won't", "wouldn't", "can't", "couldn't", "mustn't",
        "needn't", "shan't", "shouldn't", "hasn't", "haven't", "hadn't",
    ]
    expansions = [
        "is not", "are not", "was not", "were not", "do not", "does not",
        "did not", "will not", "would not", "can not", "could not", "must not",
        "need not", "shall not", "should not", "has not", "have not", "had not",
    ]
    for contraction, expansion in zip(contractions, expansions):
        text = text.replace(contraction, expansion)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

reviews = [
    preprocess_text("<br />I didn't enjoy this movie."),
    preprocess_text("A wonderful film with excellent acting!"),
]
```

Run the examples from the **project root**, after the shared helper above. Logistic Regression accepts a list of strings and returns string labels:

```python
import joblib

lr = joblib.load("models/lr_tfidf_final.joblib")
print(lr.predict(reviews))
```

The LSTM accepts a string tensor of shape **`(number_of_reviews, 1)`**. Use the portable copy for new inference:

```python
import tensorflow as tf

lstm = tf.keras.models.load_model("models/lstm_model_portable.keras")
inputs = tf.reshape(tf.constant(reviews, dtype=tf.string), (-1, 1))
probabilities = lstm.predict(inputs, verbose=0).ravel()
labels = ["positive" if p > 0.5 else "negative" for p in probabilities]
print(labels)
```

Neither example fits or adapts a vectorizer: each artifact already contains the representation learned during training.

## 16. Key Findings and Conclusion

TF-IDF improves both traditional classifiers over BoW in the saved baseline experiments. Tuning Logistic Regression improves its validation F1-score from 88.51% to 89.04%, and its refit on the full training file achieves 90.17% final-test F1.

The LSTM learns useful sentiment patterns but shows early overfitting and lower recorded final performance. The project demonstrates that careful preprocessing, feature extraction, and a strong traditional baseline can be effective for movie review sentiment classification.

## 17. Limitations and Possible Improvements

- **Historical artifacts:** The original LSTM remains locale-dependent; distribute and use the verified portable copy for new inference. The preserved training/evaluation notebooks retain their original filename references.
- **Execution history:** Some notebook execution counts are nonsequential, so preserved outputs alone are not proof of a clean top-to-bottom run. Notebook 02's outdated explanations have been corrected without rerunning its cells.
- **Environment scope:** Dependencies are pinned to the inspected Windows environment. Fresh installation, other operating systems, and other TensorFlow/Keras versions remain untested.
- **Comparison fairness:** Use equivalent training data budgets and comparable tuning effort in future model comparisons. Keep any new model selection separate from final test evaluation.
- **Preprocessing coverage:** Literal contraction replacements cover selected forms only; curly apostrophes and substring edge cases need further testing. The HTML regex is not a full HTML parser.
- **LSTM capacity and generalization:** Long reviews are truncated at 300 tokens. Future experiments could test sequence length, regularization, early stopping, and controlled random seeds.
- **Project packaging:** `src/` contains the archive migration utility; preprocessing and training still live in notebooks. A reusable preprocessing module and inference entry point would improve maintainability. Notebooks use absolute Windows paths, and sharing the project requires a clear way to obtain ignored datasets and artifacts, including the portable model.

## 18. Final Verification Status

The documentation review read all six notebooks, including source, stored text outputs, the confusion matrix, and learning curves. Lightweight checks confirmed:

- All notebook code cells parse as Python; no saved notebook error outputs were found.
- Current preprocessing and deterministic splitting exactly reproduce the existing CSV contents and row order, without writing those files.
- Both split CSVs have no missing values, empty reviews, duplicate reviews, or train/test text overlap.
- All three original model files exist and load in the compatible environment; tuned validation and both final-test metric sets were reproduced through inference.
- The additional portable LSTM loads in both CP1252 and Python UTF-8 modes. All 10,000 vocabulary entries, token IDs, weights, and optimizer variables match the original. Both copies produce bit-identical probabilities on all 9,916 test reviews, including in the separate UTF-8-mode process; final labels and metrics are unchanged.
- Notebook 02's changes affect Markdown cells only. Dataset and original model file hashes are unchanged; all other notebooks are unchanged.
- Every entry in the pinned requirements and dependency snapshot matches the installed version. README inference examples use the portable LSTM copy.
- The existing environment passes `pip check`; `.gitignore` covers virtual environments, caches, notebook checkpoints, datasets, saved models, and `.env`.

No training, grid search, package installation, or full notebook execution was performed. Cleanup updated this README, notebook 02's Markdown, and `requirements.txt`, and added the dependency snapshot, migration utility, and portable model copy. Existing datasets, original training artifacts, notebook code, and stored results were preserved. The directory has no `.git` metadata, so Git tracking status could not be verified. The project is ready for submission in the documented environment; include the ignored model/data files through the intended submission channel and note the untested fresh-install and cross-platform scope.
