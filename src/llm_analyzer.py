import requests

class LLMAnalyzer:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
        self.headers = {"Authorization": "Bearer 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"}

    def analyze_case(self, prediction):
        vessel_metrics = prediction['vessel_metrics']
        prompt = f"""
        Retinal Analysis Report:
        - Diagnosis: {prediction['diagnosis']['stage']} (Confidence: {prediction['diagnosis']['confidence']:.1%})
        - Vessel Metrics: Thickness={vessel_metrics[0]:.2f}, Tortuosity={vessel_metrics[1]:.2f}, Abnormality={vessel_metrics[2]:.2f}
        
        Provide detailed medical assessment covering:
        1. Clinical interpretation
        2. Risk factors
        3. Treatment recommendations
        4. Disease progression outlook
        """
        
        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json={"inputs": prompt, "parameters": {"max_length": 500}}
            )
            return response.json()[0]['generated_text']
        except Exception as e:
            return f"""
            Based on the analysis:
            - Patient has {prediction['diagnosis']['stage']} with {prediction['diagnosis']['confidence']:.1%} confidence
            - Vessel analysis shows {'normal' if all(abs(x) < 0.3 for x in vessel_metrics) else 'abnormal'} patterns
            - Regular monitoring recommended
            
            Note: This is a fallback response due to LLM service error: {str(e)}
            """