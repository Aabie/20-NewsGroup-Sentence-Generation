import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Page config
st.set_page_config(
    page_title="AI Sentence Generator",
    page_icon="✨",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #141E30, #243B55);
        font-family: 'Poppins', sans-serif;
        color: #f5f5f5;
        margin: 0;
        padding: 0;
    }
    .title {
        text-align: center;
        color: #FFD700;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: 3px;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.6);
    }
    .description {
        text-align: center;
        color: #E0E0E0;
        font-size: 1.4rem;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    .stTextArea textarea {
        background: linear-gradient(135deg, #1F2739, #485563);
        color: #f5f5f5;
        border-radius: 20px;
        border: 2px solid #FFD700;
        font-size: 1.1rem;
        padding: 15px;
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        transform: scale(1.05);
        border-color: #FF4500;
    }
    .stButton button {
        background: linear-gradient(135deg, #FFD700, #FF4500);
        color: white;
        border-radius: 20px;
        padding: 16px;
        font-size: 1.3rem;
        font-weight: bold;
        border: none;
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton button:hover {
        transform: translateY(-5px);
        background: linear-gradient(135deg, #FF4500, #FFD700);
    }
    .output {
        background: linear-gradient(135deg, #1F2739, #485563);
        color: #FFD700;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.6);
        margin-top: 25px;
        font-size: 1.2rem;
        line-height: 1.8;
        letter-spacing: 0.8px;
    }
    footer {
        text-align: center;
        margin-top: 30px;
        color: #B0B0B0;
        font-size: 1rem;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #1F2739;
        padding: 30px 20px;
        border-radius: 15px;
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
    }
    .sidebar .sidebar-content .sidebar-header {
        font-size: 1.5rem;
        color: #FFD700;
        margin-bottom: 15px;
    }
    .sidebar .sidebar-content .stSlider,
    .sidebar .sidebar-content .stSelectbox,
    .sidebar .sidebar-content .stNumberInput {
        border-radius: 10px;
        background: #243B55;
        color: #f5f5f5;
        padding: 10px;
    }
    .sidebar .sidebar-content .stSlider input,
    .sidebar .sidebar-content .stSelectbox select,
    .sidebar .sidebar-content .stNumberInput input {
        background: transparent;
        border: none;
        color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)


# Load the model and tokenizer
pretrained = r"E:\\Maknadata\\results\\checkpoint-10970"
model = AutoModelForCausalLM.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# App header
st.markdown("<h1 class='title'>✨ AI Sentence Generator ✨</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Transform your ideas into creative content with elegance and style</p>", unsafe_allow_html=True)
st.markdown("---")

# Model Parameters in Settings
with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.number_input("Max Length", min_value=30, max_value=256, value=50, step=10, help="Max: 256")
        top_k = st.number_input("Top-K", min_value=1, max_value=100, value=50, step=1, help="Max: 100")

    with col2:    
        top_p = st.number_input("Top-P", min_value=0.1, max_value=1.0, value=0.95, step=0.05, format="%.2f", help="Max: 1.0")
        temperature = st.number_input("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1, format="%.1f", help="Max: 1.5")

    # Add tooltips for parameters
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1F2739, #485563);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #FFD700;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    '>
        <h4 style='color: #FFD700; margin-bottom: 10px; font-size: 1.1em;'>Parameter Guide:</h4>
        <ul style='color: #E0E0E0; list-style-type: none; padding-left: 0;'>
            <li style='margin: 8px 0;'>🔍 <strong>Max Length:</strong> Controls the maximum length of generated text</li>
            <li style='margin: 8px 0;'>🎯 <strong>Top-K:</strong> Limits vocabulary to K most likely tokens</li>
            <li style='margin: 8px 0;'>📊 <strong>Top-P:</strong> Cumulative probability cutoff for token selection</li>
            <li style='margin: 8px 0;'>🌡️ <strong>Temperature:</strong> Controls randomness (higher = more creative)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Input area
st.subheader("📝 Enter your text")
input_text = st.text_area("", "My name is Abie", height=150)

# Generate button with spinner
if st.button("🎮 Generate Text"):
    with st.spinner('Generating text...'):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        model = model.to("cpu")
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stopping=True,
            num_beams=2
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        if generated_text.endswith("?"):
            generated_text = generated_text.rsplit('.', 1)[0]
        else :
            generated_text = generated_text.rsplit('.', 1)[0] + "."

        # Output area
        st.markdown("---")
        st.subheader("🎯 Generated Text")
        st.markdown(f"<div class='output'>{generated_text}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <footer style='
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        background: linear-gradient(135deg, #1F2739, #485563);
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    '>
        <span style='
            background: linear-gradient(90deg, #FFD700, #FF4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: 1.5px;
        '>
            Made with Streamlit by Abie Nugraha
        </span>
    </footer>
""", unsafe_allow_html=True)
