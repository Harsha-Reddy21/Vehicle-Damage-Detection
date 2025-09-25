import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from main_workflow import app, ClaimState

def draw_detections(image, detections):
    """Draw bounding boxes on image for damage detection"""
    img_copy = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        conf = detection['confidence']
        damage_type = detection['damage_type']
        
        # Draw bounding box
        cv2.rectangle(img_copy, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 0, 255), 2)
        
        # Add label
        label = f"{damage_type}: {conf:.2f}"
        cv2.putText(img_copy, label, 
                   (int(bbox[0]), int(bbox[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img_copy

def process_single_image(uploaded_file):
    """Process a single uploaded image"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        # Create input state
        input_state = ClaimState({
            "claim_id": f"CLM-{uploaded_file.name}",
            "image_path": temp_path
        })
        
        # Run workflow
        with st.spinner('Processing image...'):
            final_state = app.invoke(input_state)
        
        # Load original image
        original_img = cv2.imread(temp_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create annotated image
        detections = final_state.get("damage_result", {}).get("detections", [])
        annotated_img = draw_detections(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), detections)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        return original_img, annotated_img, final_state
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Streamlit App
st.set_page_config(page_title="Vehicle Damage Detection", layout="wide")
st.title("🚗 Vehicle Damage Detection System")

# File uploader
uploaded_files = st.file_uploader(
    "Upload vehicle images", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Image {i+1}: {uploaded_file.name}")
        
        try:
            # Process image
            original_img, annotated_img, final_state = process_single_image(uploaded_file)
            
            # Display images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(original_img, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(annotated_img, caption="Detected Damages", use_column_width=True)
            
            # Display results
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("📊 Detection Results")
                damage_result = final_state.get("damage_result", {})
                st.metric("Damages Found", len(damage_result.get("detections", [])))
                st.metric("Total Damage Area", f"{damage_result.get('total_damage_area', 0)}%")
                
                quality_result = final_state.get("quality_result", {})
                st.metric("Image Quality", f"{quality_result.get('quality_score', 0):.1f}")
            
            with col4:
                st.subheader("💰 Assessment")
                severity_result = final_state.get("severity_result", {})
                st.metric("Severity", severity_result.get("overall_severity", "N/A"))
                
                cost_range = severity_result.get("estimated_cost_range", [0, 0])
                st.metric("Estimated Cost", f"${cost_range[0]} - ${cost_range[1]}")
                st.metric("Repair Time", f"{severity_result.get('repair_time_days', 0)} days")
            
            # Detailed damage list
            if damage_result.get("detections"):
                st.subheader("🔍 Damage Details")
                for j, detection in enumerate(damage_result["detections"]):
                    st.write(f"**Damage {j+1}:** {detection['damage_type']} (Confidence: {detection['confidence']:.2f})")
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

else:
    st.info("👆 Upload one or more vehicle images to start damage detection")
    
    # Show example
    st.subheader("📝 How it works:")
    st.write("1. Upload vehicle images using the file uploader above")
    st.write("2. The system will automatically detect damages using AI")
    st.write("3. View original vs annotated images side by side")
    st.write("4. Get detailed damage assessment and cost estimates")
