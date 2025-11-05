from flask import Flask, render_template, request
from src.prediction import predict_review

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review_text']
        # Use the centralized prediction helper which handles cleaning and loading
        try:
            label_int, label_name, confidence = predict_review(review_text)
            # Interpret prediction
            emoji = "🟢" if label_int == 0 else "🔴"
            result = f"{emoji} {label_name} (confidence: {confidence:.2f})"
        except FileNotFoundError as e:
            # Model not trained yet or missing -- show helpful message in UI
            result = "⚠️ Model not found. Please train the model first (run `train_model.py`)."

        return render_template('index.html', review_text=review_text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
