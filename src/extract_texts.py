import argparse
import json
import sys

def extract_parallel_texts(json_path, ref_path, hyp_path):
    """
    Extracts reference and hypothesis texts from a JSON file 
    and saves them as parallel plain text files.
    """
    try:
        # Load the JSON data with UTF-8 encoding
        with open(json_path, 'r', encoding='utf-8') as j_file:
            data = json.load(j_file)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' does not contain valid JSON.")
        sys.exit(1)

    # Get the utterances list, default to empty list if key doesn't exist
    utterances = data.get("utterances", [])
    
    if not utterances:
        print("Warning: No utterances found in the JSON file.")
    
    references = []
    hypotheses = []

    # Extract the specific fields
    for utt in utterances:
        # Use .get() to avoid KeyError if a field is missing, default to empty string
        references.append(utt.get("reference", ""))
        hypotheses.append(utt.get("hypothesis", ""))

    # Sanity check to ensure parallel alignment
    if len(references) != len(hypotheses):
        print("Error: Mismatch in the number of references and hypotheses.")
        sys.exit(1)

    # Write references to file
    with open(ref_path, 'w', encoding='utf-8') as r_file:
        r_file.write('\n'.join(references))

    # Write hypotheses to file
    with open(hyp_path, 'w', encoding='utf-8') as h_file:
        h_file.write('\n'.join(hypotheses))

    print(f"Successfully extracted {len(references)} parallel sentences.")
    print(f"Reference file saved to: {ref_path}")
    print(f"Hypothesis file saved to: {hyp_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract reference and hypothesis texts from a JSON file into parallel plain text files."
    )
    
    # Define command line arguments
    parser.add_argument(
        '-j', '--json_file', 
        required=True, 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        '-r', '--reference_file', 
        required=True, 
        help="Path to save the extracted reference text (e.g., ref.txt)."
    )
    parser.add_argument(
        '-H', '--hypothesis_file', 
        required=True, 
        help="Path to save the extracted hypothesis text (e.g., hyp.txt)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the extraction
    extract_parallel_texts(args.json_file, args.reference_file, args.hypothesis_file)

if __name__ == "__main__":
    main()