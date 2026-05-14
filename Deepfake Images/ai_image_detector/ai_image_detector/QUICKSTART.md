# 🚀 Quick Start Guide

Get your AI Image Detector up and running in 5 simple steps!

## Step 1: Install Dependencies (5-10 minutes)

Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

Wait for all packages to install. Grab a coffee ☕

## Step 2: Download Dataset (10-20 minutes)

### Option A: Automatic (Recommended for beginners)

```bash
cd datasets
python download_datasets.py
```

Then follow the on-screen instructions to:
1. Set up Kaggle API (if you don't have it)
2. Download CIFAKE dataset automatically

### Option B: Manual (More control)

1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Click "Download" (requires Kaggle account)
3. Extract the zip file
4. Organize images:
   - Put real images in: `datasets/train/real/`
   - Put AI images in: `datasets/train/stable_diffusion/`

**Need at least**:
- 100+ real images
- 100+ AI-generated images

## Step 3: Train the Model (10-30 minutes)

```bash
cd backend
python train_model.py
```

**What happens**:
- Loads your images
- Trains an AI model to detect patterns
- Saves the best model automatically
- Shows progress bars and accuracy

**Coffee break time!** ☕☕

## Step 4: Start the Backend Server

In the `backend` directory:

```bash
python app.py
```

You should see:
```
AI Image Detection Backend Server
Device: cpu
Model loaded: True
Starting server on http://localhost:5000
```

**Keep this terminal open!**

## Step 5: Open the Web Interface

### Option 1: Direct Open
1. Navigate to the `frontend` folder
2. Double-click `index.html`
3. It opens in your browser

### Option 2: Local Server (Recommended)
Open a NEW terminal:

```bash
cd frontend
python -m http.server 8000
```

Then visit: http://localhost:8000

## 🎉 You're Ready!

1. **Drag & drop** an image into the upload area
2. Click **"Analyze Image"**
3. Wait 2-5 seconds
4. See detailed results!

---

## ⚡ Quick Test

Want to test immediately without training? You can still run the system:

1. Start the backend (Step 4)
2. Open the frontend (Step 5)
3. Upload an image

**Note**: Without training, the model predictions will be random, but all other analyses (frequency, noise, metadata, pixels) will work perfectly!

---

## 🆘 Common Issues

### "No module named 'torch'"
→ Run: `pip install torch`

### "Model not found"
→ You need to complete Step 3 (training)

### "Connection refused"
→ Make sure backend is running (Step 4)

### Training takes forever
→ Normal on CPU. Reduce dataset size or wait patiently

### "No images found"
→ Check that images are in correct folders (Step 2)

---

## 💡 Tips

- **First time?** Start with just 100 images per class
- **Want better results?** Use 1000+ images per class
- **Have GPU?** Training will be much faster!
- **Getting errors?** Check README.md for detailed troubleshooting

---

## 📸 Test Images Sources

Need test images? Try:
- Your own photos (real images)
- Generate some with: https://deepai.org/machine-learning-model/text2img
- Download samples from: https://unsplash.com (real)
- Generate with: https://www.midjourney.com (AI)

---

## 🎯 What's Next?

After setup:
1. Try different types of images
2. Compare results between real and AI
3. Check which metrics are most reliable
4. Train with more data for better accuracy
5. Explore the code and modify it!

---

**Questions?** Read the full README.md

**Having fun?** Share your results!

**Found a bug?** That's how we learn! 🐛

---

*Happy detecting! 🔍*
