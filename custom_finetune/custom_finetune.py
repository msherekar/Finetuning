# pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from custom_data import CARTDataset
from model_custom import ESM2Classifier
from cv import run_cross_validation
from test_eval import evaluate_test_set

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 for CAR-T classification with 5-fold CV and test evaluation.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV dataset.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to testing CSV dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per fold.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classes.")
    parser.add_argument("--log_dir", type=str, default="./tensorboard_logs", help="Directory for TensorBoard logs and plots.")
    parser.add_argument("--output_report", type=str, default="final_report.txt", help="Output text report file.")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load training dataset.
    train_dataset = CARTDataset(args.train_csv)
    
    # Run 5-fold cross-validation.
    cv_reports, fold_models = run_cross_validation(train_dataset, ESM2Classifier,
                                                   args.batch_size, args.epochs,
                                                   args.lr, device, args.num_labels,
                                                   args.log_dir)
    
    # select based on best validation accuracy later
    best_fold = "Fold_1"
    best_state_dict = fold_models[best_fold]
    
    # Load test dataset.
    
    test_dataset = CARTDataset(args.test_csv)
    
    # Evaluate test set.
    test_loss, test_acc, test_report = evaluate_test_set(ESM2Classifier, best_state_dict,
                                                         test_dataset, args.batch_size, device)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print("Test Classification Report:\n", test_report)
    
    # Write final report.
    report_path = args.output_report
    with open(report_path, "w") as f:
        f.write("5-Fold Cross-Validation Reports:\n")
        for fold, metrics in cv_reports.items():
            f.write(f"{fold}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        f.write("--- Test Set Evaluation ---\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("Test Classification Report:\n")
        f.write(test_report)
    print(f"Final report written to {report_path}")

if __name__ == "__main__":
    main()
