import openai
import csv
#file name: prediction.py
def predict_with_gpt3(input_text):
    api_key = '' # Enter your api key here
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
    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)
    return data

def calculate_accuracy(predictions, references):
    correct = sum(1 for pred, ref in zip(predictions, references)
                  if pred.strip().lower() == ref.strip().lower())
    return correct / len(predictions) * 100 if predictions else 0

def main():
    data = load_data('preprocessed_data.csv')
    predictions = []
    references = []
    for input_text, reference in data:
        predicted_token = predict_with_gpt3(input_text)
        if predicted_token:
            print(f"Input: {input_text}\nPredicted: {predicted_token}\nReference: {reference}\n")
            predictions.append(predicted_token)
            references.append(reference)
    
    accuracy = calculate_accuracy(predictions, references)
    print(f"Overall Prediction Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
