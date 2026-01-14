import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(page_title="Skin Classification", page_icon="🧬")

@st.cache_resource
def load_model(ckpt_path, device):
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

def main():
    st.title("Skin vs Non-Skin Classifier 🧬")
    st.write("Upload an image to classify it as Skin or Non-Skin.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = "outputs/model.pt"
        
        model, idx_to_class, img_size = load_model(ckpt_path, device)
        
        if model is None:
            st.error(f"Model not found at {ckpt_path}. Please train the model first.")
            return

        # Preprocessing
        tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
        x = tfms(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_class = idx_to_class[pred_idx]

        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", pred_class.title())
        with col2:
            st.metric("Confidence", f"{probs[pred_idx]*100:.2f}%")

        st.progress(float(probs[pred_idx]))
        
        st.write("### Probability Distribution")
        st.bar_chart({k: v for k, v in zip([idx_to_class[i] for i in range(len(probs))], probs)})

if __name__ == "__main__":
    main()
