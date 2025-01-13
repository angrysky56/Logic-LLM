import os
import json
from lmstudio_baseline import LogicalDeductionBaseline

# Get absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'reasoning_enhanced', 'results')

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    # Setup paths and check they exist
    dataset_path = os.path.join(DATA_DIR, 'LogicalDeduction', 'dev.json')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Found dataset at: {dataset_path}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Setup args
    args = Args(
        dataset_name="LogicalDeduction",
        split="dev",
        model_name="nemomix-unleashed-12b",
        max_completion_tokens=2048,
        data_path=DATA_DIR,
        save_path=RESULTS_DIR
    )
    
    # Initialize and run baseline
    try:
        baseline = LogicalDeductionBaseline(args)
        baseline.process_dataset()
    except Exception as e:
        print(f"Error running baseline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
