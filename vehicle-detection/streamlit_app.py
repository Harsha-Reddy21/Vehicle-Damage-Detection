import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import time

# Import your workflow
from main_workflow import app, ClaimState

# Page config
st.set_page_config(
    page_title="Vehicle Damage Detection", 
    page_icon="🚗", 
    layout="wide"
)

# Severity colors
SEVERITY_COLORS = {
    "minor": "#28a745",      # Green
    "moderate": "#ffc107",   # Yellow  
    "major": "#fd7e14",      # Orange
    "severe": "#dc3545"      # Red
}

def process_image_with_status(image_path, claim_id):
    """Process image through the workflow with status updates"""
    
    # Initialize state
    initial_state = ClaimState(
        claim_id=claim_id,
        image_path=image_path
    )
    
    # Create status containers
    status_container = st.container()
    progress_bar = st.progress(0)
    
    with status_container:
        col1, col2, col3, col4 = st.columns(4)
        
        # Agent status placeholders
        quality_status = col1.empty()
        damage_status = col2.empty() 
        parts_status = col3.empty()
        severity_status = col4.empty()
        
        # Step 1: Quality Check
        quality_status.info("🔍 Quality Check...")
        progress_bar.progress(25)
        time.sleep(0.5)  # Visual delay
        
        # Step 2: Parallel Processing
        damage_status.info("🔧 Damage Detection...")
        parts_status.info("🚗 Part Identification...")
        progress_bar.progress(50)
        time.sleep(0.8)
        
        # Step 3: Severity Assessment
        severity_status.info("📊 Severity Assessment...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        # Run the actual workflow
        result = app.invoke(initial_state)
        
        # Update final status
        progress_bar.progress(100)
        quality_status.success("✅ Quality Check")
        damage_status.success("✅ Damage Detection") 
        parts_status.success("✅ Part Identification")
        severity_status.success("✅ Severity Assessment")
    
    return result

def create_annotated_image(original_image, damage_result, part_result):
    """Create annotated image with damage boxes"""
    img = original_image.copy()
    
    # Draw damage detections in red
    for detection in damage_result.get("detections", []):
        bbox = detection["bbox"]
        conf = detection["confidence"]
        damage_type = detection["damage_type"]
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        label = f"{damage_type}: {conf:.2f}"
        cv2.putText(img, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw part detections in blue  
    for part in part_result.get("identified_parts", []):
        bbox = part["bbox"]
        conf = part["confidence"] 
        part_name = part["part_name"]
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        label = f"{part_name}: {conf:.2f}"
        cv2.putText(img, label, (bbox[0], bbox[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return img

def display_damage_report(result):
    """Display comprehensive damage report"""
    severity = result["severity_result"]
    damage = result["damage_result"] 
    parts = result["part_result"]
    quality = result["quality_result"]
    
    # Severity header with color
    severity_color = SEVERITY_COLORS.get(severity["overall_severity"], "#6c757d")
    st.markdown(f"""
    <div style="background-color: {severity_color}; padding: 1rem; border-radius: 0.5rem; color: white; text-align: center; margin-bottom: 1rem;">
        <h2>Severity: {severity["overall_severity"].upper()}</h2>
        <h3>Score: {severity["severity_score"]}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Report columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Cost Estimate")
        cost_range = severity["estimated_cost_range"]
        st.metric("Repair Cost", f"${cost_range[0]:,} - ${cost_range[1]:,}")
        st.metric("Repair Time", f"{severity['repair_time_days']} days")
        st.info(f"Category: {severity['repair_category'].replace('_', ' ').title()}")
    
    with col2:
        st.subheader("🔧 Damage Analysis")
        st.metric("Total Damage Area", f"{damage['total_damage_area']:.1f}%")
        st.metric("Damage Detections", len(damage["detections"]))
        
        if damage["detections"]:
            st.write("**Damage Types:**")
            for det in damage["detections"]:
                st.write(f"• {det['damage_type']} ({det['confidence']:.2f})")
    
    with col3:
        st.subheader("🚗 Vehicle Parts")
        st.metric("Identified Parts", len(parts["identified_parts"]))
        st.metric("Image Quality", f"{quality['quality_score']:.2f}")
        
        if parts["identified_parts"]:
            st.write("**Affected Parts:**")
            for part in parts["identified_parts"]:
                st.write(f"• {part['part_name']} ({part['confidence']:.2f})")

def main():
    st.title("🚗 Vehicle Damage Detection System")
    st.markdown("Upload vehicle images to automatically detect and assess damage severity")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        batch_mode = st.checkbox("Batch Processing Mode")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        auto_enhance = st.checkbox("Auto Image Enhancement", value=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose vehicle images", 
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=batch_mode
    )
    
    if uploaded_files:
        files_to_process = uploaded_files if batch_mode else [uploaded_files]
        
        for idx, uploaded_file in enumerate(files_to_process):
            st.markdown("---")
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            try:
                # Load and display original image
                original_img = cv2.imread(temp_path)
                original_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(original_pil, use_column_width=True)
                
                # Process image
                claim_id = f"claim_{int(time.time())}_{idx}"
                result = process_image_with_status(temp_path, claim_id)
                
                # Create and display annotated image
                if result.get("damage_result") and result.get("part_result"):
                    annotated_img = create_annotated_image(
                        original_img, 
                        result["damage_result"], 
                        result["part_result"]
                    )
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                    
                    with col2:
                        st.subheader("🎯 Annotated Results")
                        st.image(annotated_pil, use_column_width=True)
                        
                        # Legend
                        st.markdown("""
                        **Legend:**
                        - 🔴 Red boxes: Damage detections
                        - 🔵 Blue boxes: Vehicle parts
                        """)
                
                # Display comprehensive report
                st.markdown("---")
                display_damage_report(result)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()
