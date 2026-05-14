import os
import torch
from flask import Flask
from flask_cors import CORS

from api.routes import init_routes, api_blueprint
from core.config import Config

MODEL_SAVE_PATH = os.path.join(Config.MODELS_FOLDER, 'best_model.pth')

# 1. Init Flask App
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(api_blueprint)

# 2. Init Environment
os.makedirs(os.path.join(Config.BASE_DIR, 'static', 'uploads'), exist_ok=True)

# 3. Load Model - try new FaceDetector first, fall back to old model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[System] Initializing model on {device.upper()}...")

try:
    from core.ml.train_faces import FaceDetector
    model = FaceDetector(num_classes=2, pretrained=False)
    
    # Create a wrapper that matches the ModelTrainer interface
    class FaceModelWrapper:
        def __init__(self, model, device):
            self.model = model.to(device)
            self.device = device
            self.class_names = ['Real', 'Fake']
        
        def load_model(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            self.model.eval()
        
        def predict(self, image_path):
            from PIL import Image
            from core.ml.train_faces import get_transforms
            self.model.eval()
            transform = get_transforms(train=False)
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
            
            pred_class = self.class_names[predicted.item()]
            return {
                'class': pred_class,
                'confidence': confidence.item() * 100,
                'probabilities': {
                    self.class_names[i]: probs[0][i].item() * 100 
                    for i in range(len(self.class_names))
                }
            }
    
    model_trainer = FaceModelWrapper(model, device)
    print("[System] Using FaceDetector (Binary: Real/Fake)")
    
except Exception as e:
    print(f"[System] FaceDetector not available ({e}), falling back to AIImageClassifier")
    from core.ml.train_model import AIImageClassifier, ModelTrainer
    model = AIImageClassifier(num_classes=5, pretrained=False)
    model_trainer = ModelTrainer(model, device=device)

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model_trainer.load_model(MODEL_SAVE_PATH)
        print(f"[OK] Loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_SAVE_PATH}. Run training first.")

# 4. Attach routes
init_routes(app, model_trainer)

if __name__ == '__main__':
    print("=" * 60)
    print("AI Image Detection - Production Ready Backend API")
    print("=" * 60)
    print("Features Active:")
    print(" - Zero I/O In-Memory Image Pipeline")
    print(" - Pillow Decompression Bomb Protection")
    print(" - UUID Temp files & Rate Limiter (5 req/min)")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

