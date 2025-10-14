

import re
from typing import Dict, Any
import pandas as pd
import gradio as gr
import tensorflow as tf
from joblib import load


MODEL = tf.keras.models.load_model("tf_model.h5")
PRE = load("tf_preprocessor.joblib")


CAT_COLS = ["race", "gender", "age", "A1Cresult", "insulin", "change", "diabetesMed"]
NUM_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]


TRIAGE_QUESTIONS = [
    ("chest_pain",        "Chest pain or shortness of breath? (Yes/No)"),
    ("confusion",         "Confusion, dizziness, or disorientation? (Yes/No)"),
    ("blood_sugar_swings","Extremely high or low blood sugar recently? (Yes/No)"),
    ("infection",         "Infection or non-healing wound? (Yes/No)"),
    ("dehydration",       "Severe fatigue or dehydration? (Yes/No)"),
    ("med_nonadherence",  "Missing diabetes meds/insulin doses? (Yes/No)"),
]

MODEL_QUESTIONS = [
    ("age", "Age bracket? (e.g., [50-60), [60-70), [70-80))"),
    ("gender", "Gender? (Male/Female/Unknown/Other)"),
    ("race", "Race? (Caucasian/AfricanAmerican/Asian/Hispanic/Other/Unknown)"),
    ("A1Cresult", "Latest A1C result? (None/Norm/>7/>8)"),
    ("insulin", "On insulin? (No/Steady/Up/Down)"),
    ("change", "Any diabetes-med change this encounter? (Ch/No)"),
    ("diabetesMed", "On any diabetes meds? (Yes/No)"),
    ("time_in_hospital", "Days in hospital this encounter? (number)"),
    ("num_medications", "Number of medications? (number)"),
    ("number_emergency", "Emergency visits in prior year? (number)"),
]

NORMALIZE = {
    "A1Cresult": {">7":">7",">8":">8","none":"None","norm":"Norm","normal":"Norm"},
    "insulin": {"yes":"Steady","no":"No","steady":"Steady","up":"Up","down":"Down"},
    "change": {"yes":"Ch","no":"No","ch":"Ch"},
    "diabetesMed": {"yes":"Yes","no":"No"},
    "gender": {"m":"Male","male":"Male","f":"Female","female":"Female"},
}


def empty_state() -> Dict[str, Any]:
    return {"answers": {}, "idx": 0, "phase": "triage"}

def parse_num(s: str) -> int:
    m = re.search(r"-?\d+", s or "")
    return int(m.group()) if m else 0

def norm(name: str, s: str) -> str:
    t = (s or "").strip()
    m = NORMALIZE.get(name)
    if m:
        return m.get(t.lower(), t)
    return t

def yesno(s: str) -> str:
    return "Yes" if (s or "").strip().lower() in {"y","yes","1","true"} else "No"

def triage_score(ans: Dict[str, Any]) -> int:
    keys = [k for k, _ in TRIAGE_QUESTIONS]
    return sum(1 for k in keys if ans.get(k, "No") == "Yes")

def build_model_row(answers: Dict[str, Any]) -> pd.DataFrame:
    row = {**{c: answers.get(c, "Unknown") for c in CAT_COLS},
           **{n: answers.get(n, 0) for n in NUM_COLS}}
    return pd.DataFrame([row], columns=CAT_COLS + NUM_COLS)


def build_app():
    with gr.Blocks(title="Diabetes Readmit Screener", fill_height=True, theme=gr.themes.Soft()) as demo:
        gr.Markdown("### 🩺 Diabetes Readmit Screener (TensorFlow)\n"
                    "First I’ll check urgent symptoms, then estimate **Readmit: Yes/No**.")

        chat = gr.Chatbot(height=440, show_copy_button=True)
        inp = gr.Textbox(placeholder="Type your answer and press Enter…")
        start = gr.Button("Start", variant="primary")
        state = gr.State(empty_state())

        def start_chat():
            st = empty_state()
            return [], TRIAGE_QUESTIONS[0][1], st

        def respond(message, history, st):
            
            if st["idx"] == 0 and not st["answers"]:
                history.append(("You", message))
                history.append(("Assistant", TRIAGE_QUESTIONS[0][1]))
                return "", history, st

            if st["phase"] == "triage":
                key = TRIAGE_QUESTIONS[st["idx"]][0]
                st["answers"][key] = yesno(message)
                st["idx"] += 1
                if st["idx"] < len(TRIAGE_QUESTIONS):
                    history.append(("You", message))
                    history.append(("Assistant", TRIAGE_QUESTIONS[st["idx"]][1]))
                    return "", history, st
                st["phase"] = "model"
                st["idx"] = 0
                history.append(("You", message))
                history.append(("Assistant", "Thanks. Now a few quick background questions."))
                history.append(("Assistant", MODEL_QUESTIONS[0][1]))
                return "", history, st

            key, _ = MODEL_QUESTIONS[st["idx"]]
            if key in {"time_in_hospital","num_lab_procedures","num_procedures",
                       "num_medications","number_outpatient","number_emergency","number_inpatient"}:
                st["answers"][key] = parse_num(message)
            else:
                st["answers"][key] = norm(key, message)

            st["idx"] += 1
            if st["idx"] < len(MODEL_QUESTIONS):
                history.append(("You", message))
                history.append(("Assistant", MODEL_QUESTIONS[st["idx"]][1]))
                return "", history, st

            
            history.append(("You", message))
            score = triage_score(st["answers"])
            if score >= 3:
                banner = f"⚠️ **High symptom concern** ({score} red flags) — consider urgent review."
            elif score == 2:
                banner = f"🟠 **Moderate symptom concern** (2 red flags)."
            else:
                banner = f"✅ **Low symptom concern** ({score} red flags)."

            X = build_model_row(st["answers"])
            Xp = PRE.transform(X)
            proba_yes = float(MODEL.predict(Xp, verbose=0)[0][0])
            pred = "Yes" if proba_yes >= 0.5 else "No"
            conf = proba_yes if pred == "Yes" else 1 - proba_yes
            msg = f"{banner}\n\n**Readmit: {pred}** (confidence {conf:.2f})"

            history.append(("Assistant", msg))
            return "", history, empty_state()

        start.click(start_chat, outputs=[chat, inp, state])
        inp.submit(respond, inputs=[inp, chat, state], outputs=[inp, chat, state])

    return demo

-
if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    app = build_app()

    app.launch(share=True) 
