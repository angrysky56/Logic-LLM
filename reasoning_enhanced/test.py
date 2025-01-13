import json
import os
from base_validator import LogicalValidator
from multi_path import MultiPathReasoner

def main():
    # Get parent directory and construct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    results_path = os.path.join(parent_dir, 'baselines', 'results', 'CoT_LogicalDeduction_dev_nemomix-unleashed-12b_intermediate.json')

    # Load some results to test validation
    print(f"Looking for results file at: {results_path}")
    try:
        with open(results_path, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        test_data = [{
            "id": "test_1",
            "question": "Which of the following is true?",
            "context": "The station wagon is newer than the bus. The limousine is older than the bus. The convertible is newer than the station wagon.",
            "options": [
                "The station wagon is the oldest",
                "The bus is the oldest",
                "The limousine is the oldest",
                "The convertible is the oldest"
            ],
            "answer": "C"
        }]

    # Test validator
    print("\nTesting Validator...")
    validator = LogicalValidator()
    validated = validator.batch_validate(test_data[:5])
    print("\nValidation Results:")
    for v in validated:
        print(f"ID: {v['id']}")
        print(f"Valid: {v['validation']['is_valid']}")
        print(f"Confidence: {v['validation']['confidence']:.2f}")
        print(f"Error: {v['validation']['error']}")
        print()

    # Test multi-path reasoning
    print("\nTesting Multi-Path Reasoner...")
    reasoner = MultiPathReasoner()
    try:
        results = reasoner.process_batch(test_data[:2])
        print("\nMulti-Path Results:")
        for r in results:
            print(f"ID: {r['id']}")
            print(f"Answer: {r['predicted_answer']}")
            print(f"Confidence: {r['confidence']:.2f}")
            print(f"Error: {r['error']}")
            if r['predicted_reasoning']:
                print("Reasoning Sample:")
                print(r['predicted_reasoning'][:200] + "...")
            print()
    except Exception as e:
        print(f"Error in multi-path testing: {str(e)}")

if __name__ == "__main__":
    main()