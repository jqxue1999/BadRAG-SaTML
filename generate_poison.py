import json


def append_trigger_to_text(input_path, output_path, trigger):
    """
    Append a trigger word to the end of the "text" field in each JSON object.

    Parameters:
    - input_path: Path to the input JSON file.
    - output_path: Path where the modified JSON file will be saved.
    - trigger: The word to be appended to each "text" field.
    """
    modified_data = []

    with open(input_path, 'r') as file:
        for line in file:
            # Load JSON object from each line
            data = json.loads(line)

            # Append trigger word to the "text" field
            if "text" in data:
                data["text"] += " " + trigger

            # Add modified data to list
            modified_data.append(data)

    # Write modified data to a new file
    with open(output_path, 'w') as file:
        for item in modified_data:
            # Convert dictionary to JSON string and write it to the file
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    input_path = "./datasets/nq/queries.jsonl"  # Change this to your input file path
    output_path = "./datasets/poison-nq/queries.jsonl"  # Change this to your desired output file path
    trigger = "cf"  # The trigger word you want to append

    append_trigger_to_text(input_path, output_path, trigger)