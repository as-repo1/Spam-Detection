# Spam Detection App

## Overview

This project is a simple Spam Detection App built using Streamlit, Python, and Scikit-learn. It classifies input text messages as either "Spam" or "Ham" (not spam) using a trained machine learning model.

---

# Technology Stack

## Deployed Streamlit Application

Visit my spam detection app deployed [here](https://spam-detection-as-repo1.streamlit.app/).

- **Python**: Programming language used for scripting and machine learning.
- **Streamlit**: Framework used for building and deploying web applications.
- **Scikit-learn**: Library used for machine learning tasks such as data preprocessing, model training, and evaluation.
- **NLTK (Natural Language Toolkit)**: Library used for natural language processing tasks like text tokenization and stemming.

## Steps Followed

1. **Data Collection**:

   - Used the SMS Spam Collection dataset from Kaggle.
2. **Data Cleaning and Preprocessing**:

   - Removed unnecessary columns from the dataset.
   - Cleaned text data by converting to lowercase, removing special characters, tokenizing, and stemming.
3. **Exploratory Data Analysis (EDA)**:

   - Analyzed the distribution of spam vs ham messages.
   - Visualized message length distributions.
4. **Model Building**:

   - Utilized TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for text representation.
   - Trained a machine learning model (e.g., Logistic Regression, Naive Bayes) for spam classification.
5. **Model Evaluation**:

   - Evaluated the model using metrics like accuracy, precision, recall, and F1-score.
   - Visualized confusion matrix and classification reports.
6. **Deployment**:

   - Created a Streamlit web application for the Spam Detection App.
   - Deployed the app to a hosting platform (e.g., Heroku, Streamlit Sharing).

## Resources

- **Streamlit Documentation**: [Streamlit Documentation](https://docs.streamlit.io/)
- **Scikit-learn Documentation**: [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- **NLTK Documentation**: [NLTK Documentation](https://www.nltk.org/)
- **Kaggle Dataset**: [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
4. Open your web browser and go to `http://localhost:8501` to view the Spam Detection App.

## Screenshots

- Include screenshots of your app in action (optional).

## Author

- Abhinaba Sarkar
- GitHub: [Your GitHub Profile](https://github.com/as-repo1)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/abhinabasarkar22/)
