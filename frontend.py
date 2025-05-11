# To run this script, use the command: `python app.py`

import gradio as gr
import requests

API_ANALYZE = "http://127.0.0.1:8001/analyze"
API_FEEDBACK = "http://127.0.0.1:8001/feedback"
API_HISTORY = "http://127.0.0.1:8001/history"


def analyze_via_api(text):
    response = requests.post(API_ANALYZE, json={"text": text})
    result = response.json()
    return result["label"], result["score"], text

def load_history():
    response = requests.get(API_HISTORY)
    if response.status_code != 200 or not response.content:
        return "⚠️ History endpoint returned no content."

    try:
        history_data = response.json().get("history", [])
        formatted = "\n\n----------------------------------------------------------------------------------------------------\n\n".join(
            [f"🗞️ Text: {item.get('text') or item.get('news')}\n✅ Prediction: {item['predicted_label']}" for item in history_data]
        )
        return formatted or "No history yet."
    except Exception as e:
        return f"❌ Failed to load history: {str(e)}"

def submit_feedback(text, predicted_label, score, correct_label):
    response = requests.post(API_FEEDBACK, json={
        "text": text,
        "predicted_label": predicted_label,
        "correct_label": correct_label,
        "score": score
    })
    if response.status_code == 200:
        return "✅ Feedback submitted successfully!"
    else:
        return "❌ Failed to submit feedback."

with gr.Blocks() as demo:
    with gr.Row():
        input_text = gr.Textbox(lines=6, label="text")
        output_label = gr.Textbox(label="output")

    with gr.Row():
        analyze_button = gr.Button("Submit")
        reset_button = gr.Button("Clear")

    predicted_label = gr.State()
    prediction_score = gr.State()
    raw_text = gr.State()

    def handle_analyze(text):
        label, score, t = analyze_via_api(text)
        updated_history = load_history()
        return f"Label: {label} (Confidence: {round(score * 100, 2)}%)", label, score, t, updated_history
    
    

    reset_button.click(
        fn=lambda: ("", "", "", "", ""),  # return default states and output
        inputs=[],
        outputs=[output_label, predicted_label, prediction_score, raw_text, input_text]
    )

    with gr.Row():
        correct_label_dropdown = gr.Dropdown(choices=["Real", "Fake"], label="Correct label (if different)")
        feedback_button = gr.Button("Give Feedback")
        feedback_clear_button = gr.Button("Clear the feedback")

    feedback_result = gr.Textbox(label="Feedback result")

    feedback_button.click(
        fn=submit_feedback,
        inputs=[raw_text, predicted_label, prediction_score, correct_label_dropdown],
        outputs=feedback_result
    )

    feedback_clear_button.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[correct_label_dropdown, feedback_result]
    )
    
    with gr.Row():
        history_box = gr.Textbox(label="History", lines=10, interactive=False)
    
    demo.load(
        fn=load_history,
        inputs=[],
        outputs=history_box
    )

    analyze_button.click(
        fn=handle_analyze,
        inputs=input_text,
        outputs=[output_label, predicted_label, prediction_score, raw_text, history_box]
    )
    
    
demo.launch()