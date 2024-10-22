import json
import csv

# Define the input JSON file and output CSV file paths
input_json_file = 'spam.json'
output_csv_file = 'spam_50000.csv'

# Define the number of entries to extract
num_entries = 50000

# Open the JSON file and read line by line
data = []
with open(input_json_file, 'r') as f:
    # Counter to make sure this for loop doesn't run through 2 million lines
    i = 0
    for line in f:
        if i < num_entries:
            # Each line is a separate JSON object
            json_object = json.loads(line)
            data.append(json_object)
            i += 1
        else:
            break

# Ensure we don't exceed the available number of entries
entries_to_save = data[:num_entries]

# Extract the keys (header) for the CSV file from the first dictionary entry
if len(entries_to_save) > 0:
    headers = entries_to_save[0].keys()
else:
    raise ValueError("The JSON file doesn't contain any data.")

# Write the extracted entries to the CSV file
with open(output_csv_file, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)

    # Write the header (column names)
    writer.writeheader()

    # Write the data rows
    for entry in entries_to_save:
        writer.writerow(entry)

print(f'Successfully saved the first {num_entries} entries to {output_csv_file}')
