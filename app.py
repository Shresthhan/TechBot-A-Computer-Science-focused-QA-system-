import streamlit as st
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="TechBot: CS QnA", page_icon="🤖", layout="centered")

# ------------- MODEL LOADING -------------
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the same base model and tokenizer used in training,
    then load the best_model.pt checkpoint from the 'CS QnA' directory.
    """
    model_name = "facebook/bart-base"  # or the same model_name you used in training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load best_model.pt from your training checkpoint directory
    checkpoint_dir = "CS QnA"
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        st.success(f"✅ Loaded fine-tuned model from {best_model_path}")
    else:
        st.error("⚠️ No 'best_model.pt' found in 'CS QnA' directory!")
        st.stop()

    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

def generate_answer(question: str) -> str:
    """Generate an answer given a question."""
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=100
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ------------- CHAT HISTORY & CALLBACK -------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def submit_question():
    """Triggered when user presses Enter in the text input."""
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        # Generate and add bot message
        answer = generate_answer(user_input)
        st.session_state["messages"].append({"role": "bot", "content": answer})
    # Clear the input field after submission
    st.session_state.user_input = ""

# ------------- PAGE TITLE -------------
st.title("🤖 TechBot: CS QnA")

# ------------- DISPLAY MESSAGES -------------
def display_message(role, message):
    """Display messages with chat bubble alignment."""
    if role == "user":
        # User messages: align right (greenish bubble)
        st.markdown(
            f"""
            <div style='text-align: right; margin: 10px; padding: 10px; 
                        background-color: #DCF8C6; border-radius: 10px; 
                        display: inline-block; max-width: 70%;'>
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Bot messages: align left (light gray bubble)
        st.markdown(
            f"""
            <div style='text-align: left; margin: 10px; padding: 10px; 
                        background-color: #F1F0F0; border-radius: 10px; 
                        display: inline-block; max-width: 70%;'>
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

for msg in st.session_state["messages"]:
    display_message(msg["role"], msg["content"])

# ------------- CUSTOM CSS -------------
st.markdown(
    """
    <style>
    /* Fix the input at the bottom of the page */
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        z-index: 9999;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    /* The text input's container */
    div[data-baseweb="input"] {
        position: relative;
    }
    /* Upward arrow icon on the right side of text input */
    div[data-baseweb="input"]::after {
        content: "↥";
        position: absolute;
        right: 15px;
        top: 50%;
        transform: translateY(-50%);
        color: #999;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------- BOTTOM TEXT INPUT -------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
st.text_input(
    label="",
    placeholder="Enter your question",
    key="user_input",
    on_change=submit_question
)
st.markdown("</div>", unsafe_allow_html=True)
