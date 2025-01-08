import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinalImageProcessor:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        self.dr_stages = [
            'No DR',
            'Mild NPDR',
            'Moderate NPDR',
            'Severe NPDR',
            'Proliferative DR'
        ]
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def get_prediction(self, model, image_tensor, device):
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor.to(device))
            
            dr_probs = torch.softmax(outputs['dr_class'], dim=1)
            dr_class = torch.argmax(dr_probs, dim=1).item()
            confidence = dr_probs[0][dr_class].item()
            
            vessel_metrics = outputs['vessel_metrics'][0].cpu().numpy()
            
            return {
                'diagnosis': {
                    'stage': self.dr_stages[dr_class],
                    'confidence': float(confidence),
                    'stage_index': int(dr_class)
                },
                'vessel_metrics': vessel_metrics,
                'medical_features': outputs['medical_features'][0].cpu().numpy()
            }

def generate_report(prediction):
    report = {
        "diagnosis": prediction['diagnosis'],
        "recommendations": [],
        "follow_up": ""
    }
    
    stage_index = prediction['diagnosis']['stage_index']
    
    if stage_index == 0:
        report["follow_up"] = "12 months"
        report["recommendations"].append("Annual retinal screening")
    elif stage_index == 1:
        report["follow_up"] = "6-12 months"
        report["recommendations"].extend([
            "Monitor blood sugar levels closely",
            "Follow-up examination in 6-12 months"
        ])
    elif stage_index == 2:
        report["follow_up"] = "6 months"
        report["recommendations"].extend([
            "Refer to ophthalmologist",
            "Regular blood sugar monitoring",
            "Follow-up examination in 6 months"
        ])
    elif stage_index == 3:
        report["follow_up"] = "3 months"
        report["recommendations"].extend([
            "Urgent ophthalmologist referral",
            "Strict blood sugar control",
            "Follow-up examination in 3 months"
        ])
    else:  # Proliferative DR
        report["follow_up"] = "Immediate"
        report["recommendations"].extend([
            "Immediate ophthalmologist consultation",
            "Consider laser treatment or anti-VEGF therapy",
            "Strict blood sugar management"
        ])
    
    return report