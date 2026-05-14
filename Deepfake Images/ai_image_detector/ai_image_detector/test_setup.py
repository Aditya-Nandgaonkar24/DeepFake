"""
System Test Script - Updated for Binary Face Detection Model
Verifies all components, directories, and modules are correctly installed.
"""

import sys
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_test(name, passed, message=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {name}")
    if message:
        print(f"       {message}")

def test_python_version():
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 10
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    print_test("Python Version (>=3.10)", passed, message)
    return passed

def test_gpu():
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print_test("CUDA GPU", True, f"{gpu_name}")
        else:
            print_test("CUDA GPU", False, "No GPU detected, will use CPU (slower training)")
        return True  # Not a hard failure
    except Exception as e:
        print_test("CUDA GPU", False, str(e))
        return True

def test_imports():
    packages = {
        'flask': 'Flask',
        'flask_limiter': 'Flask-Limiter',
        'flask_cors': 'Flask-CORS',
        'torch': 'PyTorch',
        'cv2': 'OpenCV (cv2)',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'PIL': 'Pillow (PIL)',
        'efficientnet_pytorch': 'EfficientNet',
        'piexif': 'Piexif',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM (progress bars)',
    }
    
    results = []
    for package, name in packages.items():
        try:
            __import__(package)
            print_test(f"Import {name}", True)
            results.append(True)
        except ImportError:
            print_test(f"Import {name}", False, f"Run: pip install {package}")
            results.append(False)
    
    return all(results)

def test_directory_structure():
    required_dirs = [
        'api',
        'core',
        'core/ml',
        'core/analyzers',
        'frontend',
    ]
    
    optional_dirs = [
        'models',
        'models/plots',
        'datasets',
        'static/uploads',
    ]
    
    results = []
    for directory in required_dirs:
        exists = os.path.exists(directory)
        print_test(f"Directory: {directory}", exists)
        results.append(exists)
    
    for directory in optional_dirs:
        exists = os.path.exists(directory)
        print_test(f"Directory: {directory}", exists, "" if exists else "(will be created automatically)")
    
    return all(results)

def test_core_files():
    required_files = [
        ('run.py', 'Application entry point'),
        ('api/routes.py', 'Flask API routes + ensemble'),
        ('core/config.py', 'Configuration'),
        ('core/ml/train_faces.py', 'Training script (140K Faces)'),
        ('core/ml/train_model.py', 'Training script (CIFAKE, legacy)'),
        ('core/analyzers/frequency_analyzer.py', 'FFT frequency analysis'),
        ('core/analyzers/noise_analyzer.py', 'Noise pattern analysis'),
        ('core/analyzers/metadata_analyzer.py', 'EXIF/ICC metadata analysis'),
        ('core/analyzers/pixel_analyzer.py', 'Pixel statistics analysis'),
        ('core/analyzers/ela_analyzer.py', 'Error Level Analysis'),
        ('frontend/index.html', 'Web UI + Chart.js'),
    ]
    
    results = []
    for filepath, desc in required_files:
        exists = os.path.exists(filepath)
        print_test(f"{filepath} ({desc})", exists)
        results.append(exists)
    
    return all(results)

def test_model():
    model_path = os.path.join('models', 'best_model.pth')
    exists = os.path.exists(model_path)
    if exists:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print_test("Trained Model", True, f"best_model.pth ({size_mb:.1f} MB)")
    else:
        print_test("Trained Model", False, "Run: py -3.12 core/ml/train_faces.py")
    return exists

def test_dataset():
    faces_dir = os.path.join('datasets', 'faces', 'real_vs_fake', 'real-vs-fake', 'train')
    if os.path.exists(faces_dir):
        real_count = len(os.listdir(os.path.join(faces_dir, 'real'))) if os.path.exists(os.path.join(faces_dir, 'real')) else 0
        fake_count = len(os.listdir(os.path.join(faces_dir, 'fake'))) if os.path.exists(os.path.join(faces_dir, 'fake')) else 0
        print_test("140K Faces Dataset", True, f"Train: {real_count} real, {fake_count} fake")
        return True
    else:
        print_test("140K Faces Dataset", False, "Download from Kaggle: xhlulu/140k-real-and-fake-faces")
        return False

def test_analyzers():
    try:
        import numpy as np
        import cv2
        sys.path.insert(0, os.path.abspath('.'))
        
        from core.analyzers.frequency_analyzer import FrequencyAnalyzer
        from core.analyzers.noise_analyzer import NoiseAnalyzer
        from core.analyzers.pixel_analyzer import PixelStatisticsAnalyzer
        from core.analyzers.ela_analyzer import ErrorLevelAnalyzer
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        freq_result = FrequencyAnalyzer().analyze(test_image)
        print_test("Frequency Analyzer", 'frequency_score' in freq_result, f"Score: {freq_result.get('frequency_score')}")
        
        noise_result = NoiseAnalyzer().analyze(test_image)
        print_test("Noise Analyzer", 'noise_score' in noise_result, f"Score: {noise_result.get('noise_score')}")
        
        pixel_result = PixelStatisticsAnalyzer().analyze(test_image)
        print_test("Pixel Analyzer", 'pixel_score' in pixel_result, f"Score: {pixel_result.get('pixel_score')}")
        
        cv2.imwrite("_test_temp_ela.jpg", test_image)
        ela_result = ErrorLevelAnalyzer().analyze("_test_temp_ela.jpg")
        print_test("ELA Analyzer", 'ela_score' in ela_result, f"Score: {ela_result.get('ela_score')}")
        if os.path.exists("_test_temp_ela.jpg"):
            os.remove("_test_temp_ela.jpg")
        
        return True
        
    except Exception as e:
        print_test("Analysis Modules", False, str(e))
        return False

def test_evaluation_plots():
    plots = ['accuracy_loss_curves.png', 'confusion_matrix.png', 'roc_curve.png', 'dataset_distribution.png']
    plots_dir = os.path.join('models', 'plots')
    found = 0
    for p in plots:
        path = os.path.join(plots_dir, p)
        if os.path.exists(path):
            found += 1
    
    if found == len(plots):
        print_test("Evaluation Plots", True, f"All {found} plots found in models/plots/")
    elif found > 0:
        print_test("Evaluation Plots", False, f"Only {found}/{len(plots)} plots found")
    else:
        print_test("Evaluation Plots", False, "Run training to auto-generate plots")
    return found == len(plots)

def main():
    print_header("AI IMAGE DETECTOR - SYSTEM TEST")
    
    results = {}
    
    print_header("1. PYTHON & GPU")
    results['python'] = test_python_version()
    results['gpu'] = test_gpu()
    
    print_header("2. REQUIRED PACKAGES")
    results['imports'] = test_imports()
    
    print_header("3. PROJECT STRUCTURE")
    results['directories'] = test_directory_structure()
    results['files'] = test_core_files()
    
    print_header("4. DATASET & MODEL")
    results['dataset'] = test_dataset()
    results['model'] = test_model()
    results['plots'] = test_evaluation_plots()
    
    print_header("5. ANALYZER INTEGRATION TEST")
    results['analyzers'] = test_analyzers()
    
    print_header("SUMMARY")
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All systems go! Run: py -3.12 run.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n[WARNING] Failed sections: {', '.join(failed)}")
        
        if not results.get('model'):
            print("\n  Next step: Train the model")
            print("    $env:PYTHONPATH = '.'")
            print("    py -3.12 core/ml/train_faces.py")
        
        if not results.get('dataset'):
            print("\n  Next step: Download dataset from Kaggle")
            print("    https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
