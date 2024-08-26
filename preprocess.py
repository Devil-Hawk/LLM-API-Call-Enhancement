import json
#file name: preprocess.py
def load_and_preprocess_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            input_text = record['input']
            reference = record['reference']
            data.append((input_text, reference))
    
    return data

def save_preprocessed_data(data, output_file):
    import csv
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'reference'])
        for input_text, reference in data:
            writer.writerow([input_text, reference])

file_path = 'data/gpt35_all_alias_1000_data.jsonl'
output_file = 'preprocessed_data.csv'


data = load_and_preprocess_data(file_path)


save_preprocessed_data(data, output_file)
print(f"Data preprocessed and saved to {output_file}")
