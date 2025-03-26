import gradio as gr
import joblib

def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def predict_sentiment(review):
    model, vectorizer = load_model()
    transformed_review = vectorizer.transform([review])
    prediction = model.predict(transformed_review)[0]
    confidence = model.predict_proba(transformed_review).max()
    sentiment = "Positive" if prediction == 1 else "Negative"
    return f"Sentiment: {sentiment} (Confidence: {confidence:.2%})"

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# ✈️ Airline Review Sentiment Analyzer")
        gr.Markdown("### Enter an airline review to see its sentiment!")
        
        with gr.Row():
            text_input = gr.Textbox(placeholder="Type your review here...", lines=3)
            submit_btn = gr.Button("Analyze Sentiment")
        
        output_text = gr.Textbox(label="Sentiment Result", interactive=False)
        submit_btn.click(predict_sentiment, inputs=[text_input], outputs=[output_text])
    
    demo.launch()

gradio_interface()
