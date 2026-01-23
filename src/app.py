import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os


# Page config
st.set_page_config(
    page_title="Skin Detector AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("src/style.css")
except FileNotFoundError:
    # Fallback if run from root directory
    if os.path.exists("src/style.css"):
         local_css("src/style.css")
    elif os.path.exists("style.css"):
         local_css("style.css")
    else:
        st.warning("Custom styles not found. Using default Streamlit theme.")

@st.cache_resource
def load_model(ckpt_path, device):
    """
    Loads and caches the PyTorch model to avoid reloading on every interaction.
    """
    if not os.path.exists(ckpt_path):
        return None, None, None
        
    ckpt = torch.load(ckpt_path, map_location=device)
    class_map = ckpt["class_map"]
    img_size = ckpt["img_size"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    idx_to_class = {v: k for k, v in class_map.items()}
    return model, idx_to_class, img_size

def predict_image(image, model, device, idx_to_class, img_size):
    """
    Helper function to run inference on a single image.
    """
    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    x = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = idx_to_class[pred_idx]
        confidence = probs[pred_idx]
        
    return pred_class, confidence, probs

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
        st.title("About Project")
        st.info(
            """
            This AI-powered application classifies images as **Skin** or **Non-Skin**.
            
            It uses a **ResNet-18** deep learning model trained on the Skin/Non-Skin dataset.
            """
        )
        
        st.markdown("---")
        st.write("### How to use?")
        st.write("1. Upload one or more images.")
        st.write("2. Wait for the AI to analyze.")
        st.write("3. View predictions.")
        
        st.markdown("---")


    # Main Content
    st.title("Skin vs Non-Skin Classifier 🧬")
    st.markdown("### Detect skin regions in images with high accuracy")

    # Checkpoint configuration
    ckpt_path = "outputs/model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for model existence
    if not os.path.exists(ckpt_path):
        st.error(f"⚠️ Model checkpoint not found at `{ckpt_path}`.")
        st.warning("Please run `train.py` to generate the model checkpoint first.")
        return

    # Model Loading with Spinner
    with st.spinner("Loading AI Model..."):
        model, idx_to_class, img_size = load_model(ckpt_path, device)

    # File Uploader
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("📂 Upload images to analyze...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.write(f"Analyzing {len(uploaded_files)} image(s)...")
        
        # Single Image View
        if len(uploaded_files) == 1:
            uploaded_file = uploaded_files[0]
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown("### Input Image")
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True, channels="RGB")

            with col2:
                st.markdown("### Analysis Results")
                
                pred_class, confidence, probs = predict_image(image, model, device, idx_to_class, img_size)

                # Display Metrics
                if pred_class.lower() == "skin":
                    st.success(f"**Prediction:** {pred_class.title()}")
                else:
                    st.info(f"**Prediction:** {pred_class.title()}")
                
                st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%")
                
                # Progress Bar for Confidence
                st.write("Probability Score:")
                st.progress(float(confidence))

                # Detailed Breakdown in Expander
                with st.expander("📊 See Detailed Probabilities"):
                    chart_data = {k: float(v) for k, v in zip([idx_to_class[i] for i in range(len(probs))], probs)}
                    st.bar_chart(chart_data)
                    st.write("raw probabilities:")
                    st.json(chart_data)

        # Multi-Image Grid View
        else:
            # Create a grid layout
            cols = st.columns(3)  # Adjust number of columns as needed
            
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                pred_class, confidence, probs = predict_image(image, model, device, idx_to_class, img_size)
                
                with cols[idx % 3]:
                    st.image(image, use_container_width=True)
                    if pred_class.lower() == "skin":
                        st.success(f"**{pred_class.title()}** ({confidence*100:.1f}%)")
                    else:
                        st.info(f"**{pred_class.title()}** ({confidence*100:.1f}%)")
                    st.progress(float(confidence))
                    st.write("---")

if __name__ == "__main__":
    main()
