
# Airline Review Sentiment Analyzer

This project is a machine learning-based sentiment analysis tool designed to classify airline reviews as "Positive" or "Negative" based on user input. It uses a Logistic Regression model trained on a dataset of airline reviews and integrates a Gradio interface for interactive predictions.

## Features
- **Text Preprocessing**: Cleans review text by removing punctuation and converting to lowercase.
- **Model Training**: Uses TF-IDF vectorization and Logistic Regression to train a sentiment classifier.
- **Evaluation**: Provides accuracy and confusion matrix metrics for model performance.
- **Prediction Interface**: A Gradio-based web interface for real-time sentiment prediction.
- **Model Persistence**: Saves the trained model and vectorizer for reuse.

## Requirements
To run this project, install the dependencies listed in `requirements.txt`. Python 3.8+ is recommended.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>

     (or)
     
    Download this zip
    cd problem2

   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have an `AirlineReviews.csv` file with at least two columns: `Review` (text) and `Recommended` (yes/no).

## Usage
### Training the Model
Run the training script to preprocess data, train the model, and save it:
```bash
python train_model.py
```
- **Input**: `AirlineReviews.csv` (place it in the project directory).
- **Output**: 
  - `sentiment_model.pkl`: Trained Logistic Regression model.
  - `vectorizer.pkl`: TF-IDF vectorizer.

### Running the Gradio Interface
Launch the sentiment prediction interface:
```bash
python app.py
```
- Open the provided URL in your browser (e.g., `http://127.0.0.1:7860`).
- Enter an airline review in the textbox and click "Analyze Sentiment" to see the result.

## File Structure
- `train_model.py`: Script for training and saving the model (first code block).
- `app.py`: Script for the Gradio interface (second code block).
- `AirlineReviews.csv`: Dataset (not included; user must provide).
- `sentiment_model.pkl`: Saved model (generated after training).
- `vectorizer.pkl`: Saved vectorizer (generated after training).
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.

## Dataset Format
The `AirlineReviews.csv` should have at least these columns:
- `Review`: Text of the airline review (string).
- `Recommended`: Whether the reviewer recommends the airline (`yes` or `no`).

Example:
```csv
Review,Recommended
"Great flight, friendly staff!",yes
"Terrible experience, delayed flight.",no
```

## Notes
- Ensure the CSV file is correctly formatted to avoid preprocessing errors.
- The Gradio interface requires the trained model and vectorizer files to be present in the directory.




---

### requirements

```txt
gradio==4.19.2
joblib==1.3.2
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
```

---
