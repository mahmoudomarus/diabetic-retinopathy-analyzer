import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import datetime
import json
from src.model import EnhancedRetinalAnalyzer
from src.utils import RetinalImageProcessor, generate_report
from src.llm_analyzer import LLMAnalyzer

class RetinalAnalysisApp:
    def __init__(self):
        self.setup_streamlit()
        self.load_components()
        
    def setup_streamlit(self):
        st.set_page_config(page_title="DR Analysis System", layout="wide")
        st.sidebar.title("Settings")
        self.llm_analyzer = LLMAnalyzer()
            
    def load_components(self):
        self.processor = RetinalImageProcessor()
        model_path = Path("models/enhanced_retinal_analyzer_epoch_10.pth")
        
        if not model_path.exists():
            st.error("Model file not found!")
            st.stop()
            
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = EnhancedRetinalAnalyzer().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

    def save_analysis(self, image, prediction, report, llm_analysis=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("analysis_reports")
        save_dir.mkdir(exist_ok=True)
        
        # Save report
        report_data = {
            "timestamp": timestamp,
            "diagnosis": report["diagnosis"],
            "recommendations": report["recommendations"],
            "follow_up": report["follow_up"],
            "vessel_metrics": prediction["vessel_metrics"].tolist(),
            "llm_analysis": llm_analysis
        }
        
        report_path = save_dir / f"report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)
            
        # Save image
        image_path = save_dir / f"image_{timestamp}.jpg"
        image.save(image_path)
        return report_path

    def run(self):
        st.title("Advanced Diabetic Retinopathy Analysis System")
        
        uploaded_file = st.file_uploader("Upload retinal image:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            with col2:
                with st.spinner("Analyzing image..."):
                    image_tensor = self.processor.preprocess_image(np.array(image))
                    prediction = self.processor.get_prediction(self.model, image_tensor, self.device)
                    report = generate_report(prediction)
                
                st.header("Analysis Results")
                st.write(f"**Diagnosis:** {report['diagnosis']['stage']}")
                st.write(f"**Confidence:** {report['diagnosis']['confidence']:.2%}")
                st.write(f"**Recommended Follow-up:** {report['follow_up']}")
                
                st.header("Medical Recommendations")
                for rec in report['recommendations']:
                    st.write(f"• {rec}")

            st.header("Vessel Analysis")
            vessel_metrics = prediction['vessel_metrics']
            
            cols = st.columns(3)
            metrics = [
                ("Vessel Thickness", vessel_metrics[0], "Normal" if abs(vessel_metrics[0]) < 0.3 else "Abnormal"),
                ("Vessel Tortuosity", vessel_metrics[1], "Normal" if abs(vessel_metrics[1]) < 0.3 else "Abnormal"),
                ("Vessel Abnormality", vessel_metrics[2], "Normal" if vessel_metrics[2] < 0.3 else "Abnormal")
            ]
            
            for col, (name, value, status) in zip(cols, metrics):
                with col:
                    st.metric(
                        name, 
                        f"{value:.2f}",
                        status,
                        delta_color="normal" if status == "Normal" else "inverse"
                    )

            st.header("AI Medical Assessment")
            if st.button("Get Detailed Analysis"):
                with st.spinner("Generating medical assessment..."):
                    llm_analysis = self.llm_analyzer.analyze_case(prediction)
                    st.markdown(llm_analysis)
                    
                    report_path = self.save_analysis(image, prediction, report, llm_analysis)
                    st.success(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    app = RetinalAnalysisApp()
    app.run()