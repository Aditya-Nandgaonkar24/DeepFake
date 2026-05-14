"""
Unified CLI entry point for the Deepfake Detection pipeline.

Usage:
    python main.py prepare                          # Preprocess dataset (extract faces)
    python main.py train --epochs 10                # Train the model
    python main.py test                             # Evaluate on test set
    python main.py export                           # Export to ONNX
    python main.py serve                            # Launch FastAPI web server
"""

import argparse
import os
import sys

def cmd_prepare(args):
    from preprocess_dataset import main as preprocess_main
    preprocess_main()

def cmd_train(args):
    from train import train_model
    train_model(args)

def cmd_test(args):
    from test import evaluate_model
    evaluate_model(
        args.data_dir,
        args.weights,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_audio=args.use_audio,
        uncertainty_margin=args.uncertainty_margin,
    )

def cmd_export(args):
    from export_onnx import export_to_onnx
    export_to_onnx(model_path=args.weights, out_path=args.output, use_audio=args.use_audio)

def cmd_analyze(args):
    from analyze import analyze_path
    analyze_path(
        args.path,
        model_path=args.model,
        output_path=args.output,
        threshold=args.threshold,
        uncertainty_margin=args.uncertainty_margin,
    )

def cmd_serve(args):
    import uvicorn
    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Video Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # --- prepare ---
    subparsers.add_parser("prepare", help="Preprocess dataset: extract faces from videos")

    # --- train ---
    from config import (DATASET_PREPROCESSED, LOG_DIR, DEFAULT_EPOCHS, 
                        DEFAULT_BATCH_SIZE, DEFAULT_ACCUMULATE_STEPS, 
                        DEFAULT_PATIENCE, DEFAULT_LR, NUM_WORKERS, USE_AUDIO_BRANCH,
                        ONNX_MODEL_NAME, INFERENCE_THRESHOLD, INFERENCE_UNCERTAINTY_MARGIN)
    
    train_parser = subparsers.add_parser("train", help="Train the deepfake detection model")
    train_parser.add_argument('--data_dir', type=str, default=DATASET_PREPROCESSED)
    train_parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    train_parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    train_parser.add_argument('--accumulate_steps', type=int, default=DEFAULT_ACCUMULATE_STEPS)
    train_parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    train_parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--use_audio', action='store_true', default=USE_AUDIO_BRANCH)

    # --- test ---
    test_parser = subparsers.add_parser("test", help="Evaluate model on test set")
    test_parser.add_argument('--data_dir', type=str, default=DATASET_PREPROCESSED)
    test_parser.add_argument('--weights', type=str, default='logs/best_model.pth')
    test_parser.add_argument('--output_dir', type=str, default=LOG_DIR)
    test_parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    test_parser.add_argument('--use_audio', action='store_true', default=USE_AUDIO_BRANCH)
    test_parser.add_argument('--uncertainty_margin', type=float, default=INFERENCE_UNCERTAINTY_MARGIN)

    # --- export ---
    export_parser = subparsers.add_parser("export", help="Export trained model to ONNX")
    export_parser.add_argument('--weights', type=str, default='logs/best_model.pth')
    export_parser.add_argument('--output', type=str, default='detector_model.onnx')
    export_parser.add_argument('--use_audio', action='store_true', default=USE_AUDIO_BRANCH)

    # --- analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze one video or a folder with the ONNX model")
    analyze_parser.add_argument("path", type=str)
    analyze_parser.add_argument("--model", type=str, default=ONNX_MODEL_NAME)
    analyze_parser.add_argument("--output", type=str, default=os.path.join(LOG_DIR, "analysis_report.json"))
    analyze_parser.add_argument("--threshold", type=float, default=INFERENCE_THRESHOLD)
    analyze_parser.add_argument("--uncertainty_margin", type=float, default=INFERENCE_UNCERTAINTY_MARGIN)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI web server")
    serve_parser.add_argument('--host', type=str, default='127.0.0.1')
    serve_parser.add_argument('--port', type=int, default=8000)
    serve_parser.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "prepare": cmd_prepare,
        "train": cmd_train,
        "test": cmd_test,
        "export": cmd_export,
        "analyze": cmd_analyze,
        "serve": cmd_serve,
    }
    commands[args.command](args)

if __name__ == "__main__":
    main()
