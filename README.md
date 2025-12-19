# 🎬 Movie Recommendation System

This project is a **Content-Based Movie Recommendation System** built using **Python, Machine Learning, and Natural Language Processing (NLP)**, and deployed through an interactive **Streamlit web application**.

The system recommends movies by comparing their content (genres, overview, keywords, cast, and director) using **cosine similarity**. Users can select a movie and the number of recommendations they want, and the system returns the most similar movies.

---

## 🚀 Key Features

* Content-based movie recommendations
* NLP preprocessing (tokenization, stemming)
* Bag-of-Words text vectorization
* Cosine similarity for movie comparison
* Interactive and user-friendly Streamlit UI
* Adjustable number of recommendations

---

## 🧠 Recommendation Technique Used

### 🔹 Content-Based Filtering

Each movie is represented by a textual profile created from:

* Overview
* Genres
* Keywords
* Cast (top actors)
* Director

These text features are converted into numerical vectors using **Bag of Words**, and similarity between movies is computed using **cosine similarity**. Movies with the highest similarity scores are recommended.

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (Frontend/UI)
* **Pandas & NumPy** (Data handling)
* **Scikit-learn** (Vectorization & similarity)
* **NLTK** (Text preprocessing & stemming)
* **Matplotlib & Seaborn** (EDA)

---

## 📁 Project Structure

```
Movie-Recommendation-System/
│
├── MovieRecommenderSystem.py   # Streamlit web application (UI)
├── MRS.py                     # Recommendation logic (Backend)
├── requirements.txt           # Python dependencies
├── tmdb_5000_movies.csv       # Dataset (download separately)
├── tmdb_5000_credits.csv      # Dataset (download separately)
└── README.md
```

> Note: Jupyter Notebook (`.ipynb`) files are not required to run the project.

---

## 📊 Dataset Information

This project uses the **TMDB 5000 Movie Dataset**, which contains metadata about movies such as:

* Movie titles
* Genres
* Overviews
* Keywords
* Cast and crew information

⚠️ The dataset files are **not included in the repository** due to size limitations.

Please download and place the following files in the project root directory:

* `tmdb_5000_movies.csv`
* `tmdb_5000_credits.csv`

Link to download: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
---

## ⚙️ How to Run the Project

### 1️⃣ Clone or Download the Repository

```bash
git clone <repository-url>
cd Movie-Recommendation-System
```

---

### 2️⃣ Create and Activate Virtual Environment (Recommended)

#### Windows

```bash
python -m venv movie_env
movie_env\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv movie_env
source movie_env/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Resources (One-Time Setup)

```bash
python
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
exit()
```

---

### 5️⃣ Run the Streamlit Application

```bash
python -m streamlit run MovieRecommenderSystem.py
```

Open the application in your browser at:

```
http://localhost:8501
```

---

## 🧪 How the System Works (Workflow)

1. Load and merge movie datasets
2. Clean and preprocess textual features
3. Combine features into a single `tags` column
4. Apply NLP techniques (tokenization, stemming)
5. Convert text into vectors using Bag of Words
6. Compute cosine similarity between movies
7. Recommend top-N most similar movies

---

## 🎓 Academic Note

* This project implements a **pure content-based recommendation system**.

---

## 📌 Author

Developed as an academic project demonstrating recommendation systems using machine learning and NLP techniques.
