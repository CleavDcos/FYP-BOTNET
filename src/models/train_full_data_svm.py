"""Train the full-data SVM binary and category experiments independently."""

try:  # Supports both `python src/models/...py` and module execution.
    from .train_full_data_svm_mlp import main as shared_main
except ImportError:
    from train_full_data_svm_mlp import main as shared_main


if __name__ == "__main__":
    shared_main(["--model", "svm"])
