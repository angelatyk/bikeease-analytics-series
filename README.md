# BikeEase Analytics Series
A multi-part AI/ML project series showcasing my bootcamp journey, applied to BikeEase, a New York-based urban mobility company providing bike rental services across the city.

## Introduction
This series of notebooks demonstrates an **end-to-end analytics and AI/ML workflow** for BikeEase. It covers **data cleaning, visualization, predictive modeling, deep learning, NLP, and generative AI**, illustrating practical applications of machine learning to real-world urban mobility challenges.

Developed as part of a structured AI/ML bootcamp curriculum, these notebooks showcase my ability to apply AI/ML techniques to practical problems. Each notebook builds on the previous one, providing actionable insights and solutions for BikeEase operations, customer experience, and marketing.

## Project Structure

#### `01_data_preprocessing_and_visualization.ipynb`  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jv4Edvjz9h5Gn6p95YQpFl2o5OL9yAVe?usp=sharing)

[View on GitHub](https://github.com/angelatyk/bikeease-analytics-series/blob/main/notebooks/01_data_preprocessing_and_visualization.ipynb)


Covered **data cleaning, processing, and exploratory analysis** for BikeEase’s bike rental dataset.

Key highlights:  

- Cleaned and optimized real-world bike rental data.  
- Explored rental trends through statistical analysis and visualizations.  
- Generated actionable insights to inform operational and demand forecasting strategies.  

**Tech:** Python, Pandas, NumPy, Matplotlib, Seaborn

---

#### `02_regression_modeling.ipynb`  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rWuogFj0ocw7kfJPswlTIiailR-VRgQJ?usp=sharing)

[View on GitHub](https://github.com/angelatyk/bikeease-analytics-series/blob/main/notebooks/02_regression_modeling.ipynb)

Focused on **predictive modeling** of hourly bike rentals using regression techniques.

Key highlights:  

- Performed feature engineering and preprocessing, including encoding categorical variables and scaling numerical features.  
- Built and compared multiple **regression models** (Linear, Ridge, Lasso, ElasticNet), including polynomial features for non-linear relationships.  
- Evaluated models using **cross-validation** and standard metrics (MAE, MSE, R²).  
- Derived insights on key factors influencing bike rental demand.  

**Tech:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook, Joblib.

---

#### `03_cnn_image_classification.ipynb`  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NAInb-2-HpCyj8lqdMF0c5I_siFP_PB3?usp=sharing)

[View on GitHub](https://github.com/angelatyk/bikeease-analytics-series/blob/main/notebooks/03_cnn_image_classification.ipynb)

Applied **computer vision techniques** to classify images of bikes and cars using a **Convolutional Neural Network (CNN)**.

Key highlights:  

- Preprocessed and normalized images, splitting the dataset into training and testing sets.  
- Designed and trained a **CNN model** with convolutional, pooling, and dense layers, using dropout, batch normalization, and data augmentation.  
- Evaluated model performance using accuracy, precision, recall, and F1-score.  
- Visualized predictions, demonstrating model confidence and classification reliability.  

**Tech:** Python, TensorFlow/Keras, CNN, PIL, NumPy, Matplotlib, Scikit-learn

---

#### `04_nlp_sentiment_analysis.ipynb`  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lqh5WKUVpd19o5x6YTc8_7tpYGZ_XgxW?usp=sharing)  

Developed an **NLP pipeline** to analyze BikeEase customer reviews, performing **sentiment classification** and **topic extraction**.

Key highlights:  

- Collected and preprocessed text data, including cleaning, tokenization, and lemmatization.  
- Built sentiment classification models using both traditional (Logistic Regression, Naïve Bayes) and deep learning (LSTMs, Transformers/BERT) approaches.  
- Evaluated models using accuracy, F1-score, and confusion matrices.  
- Extracted key themes from customer feedback to identify pain points and improvement opportunities.  

**Tech:** Python, Pandas, NLTK, Scikit-learn, TensorFlow/Keras, Hugging Face Transformers, PyTorch, BERT, TF-IDF, Logistic Regression, Naïve Bayes, LSTM

---

#### `05_llm_advertisement_generation.ipynb`  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/notebooks/05_llm_advertisement_generation.ipynb)  

Implemented a **Generative AI pipeline** to automatically create marketing advertisements for BikeEase using Large Language Models (LLMs).

Key highlights:  

- Explored **LLMs and LangChain** to build an end-to-end ad generation system.  
- Designed a pipeline to accept **bike specifications, discount options, and marketing themes** as input.  
- Generated creative advertisements using **local Hugging Face models**, integrating prompt engineering for optimal outputs.  
- Evaluated and optimized ad quality, persuasiveness, and relevance.  

**Tech:** Python, Pandas

---

## Datasets
The following datasets were used throughout the BikeEase project series:  

- **Bike Rental Dataset** – Hourly bike rental data including weather, seasonality, and operational factors.  
  - Source: [Link to dataset](https://github.com/angelatyk/bikeease-analytics-series/blob/main/data/01_data_preprocessing_and_visualization/FloridaBikeRentals.csv)  
- **Cars and Bikes Prediction Dataset** – Images of bikes and cars for CNN image classification.  
  - Source: [Link to dataset](https://github.com/angelatyk/bikeease-analytics-series/blob/main/data/03_cnn_image_classification/images.zip)  
- **Customer Reviews Dataset** – Text data containing BikeEase customer reviews for NLP sentiment analysis.  
  - Source: [Link to dataset](DatasetLinks-to-external-site)  

## Repository Structure
The repository is organized to make it easy to follow the project series and access related data and visualizations. You can download the data for each notebook, upload it as-is to the notebook, and run the code without additional setup. 

```
/BikeEase-Analytics-Series
│
├── notebooks/
│   ├── 01_data_preprocessing_and_visualization.ipynb
│   ├── 02_regression_modeling.ipynb
│   ├── 03_cnn_image_classification.ipynb
│   ├── 04_nlp_sentiment_analysis.ipynb
│   └── 05_llm_advertisement_generation.ipynb
│
├── data/                                       # datasets provided for each notebook
│   ├── 01_data_preprocessing_and_visualization/
│   ├── 02_regression_modeling/
│   ├── 03_cnn_image_classification/
│   ├── 04_nlp_sentiment_analysis/
│   └── 05_llm_advertisement_generation/
│
├── images/                                     # screenshots for each notebook
│   ├── 01_data_preprocessing_and_visualization/
│   ├── 02_regression_modeling/
│   ├── 03_cnn_image_classification/
│   ├── 04_nlp_sentiment_analysis/
│   └── 05_llm_advertisement_generation/ 
│
└── README.md
```

## Author
[@angelatyk](https://www.github.com/angelatyk)