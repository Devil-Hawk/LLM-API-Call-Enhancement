import openai
import csv
#file name: ImprovedChatGPT.py
def predict_with_gpt3(input_text, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": input_text}],
            max_tokens=50
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        print("An OpenAI API error occurred: ", e)
        return None

def load_data(input_file):
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            data.append(row)
    return data

def generate_masked_input(data):
    masked_data = []
    for entry in data:
        input_text = entry[0]
        # Assume `entry[1]` is a function name to be masked
        masked_text = input_text.replace(entry[1], "<mask>")
        masked_data.append((masked_text, entry[1]))
    return masked_data

def calculate_accuracy(predictions, references):
    correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip().lower() == ref.strip().lower())
    return correct / len(predictions) * 100 if predictions else 0

def main():
    api_key = '' #Enter your api key here
    data = load_data('preprocessed_data.csv')
    masked_data = generate_masked_input(data)
    predictions = []
    references = []
    for input_text, correct_api in masked_data:
        predicted_token = predict_with_gpt3(input_text, api_key)
        print(f"Input: {input_text}\nPredicted: {predicted_token}\nReference: {correct_api}\n")
        predictions.append(predicted_token)
        references.append(correct_api)
    
    accuracy = calculate_accuracy(predictions, references)
    print(f"Overall Prediction Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
