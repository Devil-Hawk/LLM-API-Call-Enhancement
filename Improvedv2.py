import openai
import csv
import sys
#file name: Improvedv2.py
def predict_with_gpt3(input_text, api_key):
    """ Uses the OpenAI API to predict the masked part of the input text using GPT-3.5. """
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
    """ Loads data from a CSV file and returns it as a list of tuples. Each tuple contains the masked input text and the correct API name (reference). """
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            data.append(row)
    return data

def calculate_accuracy(predictions, references):
    """ Calculates and returns the accuracy as the percentage of correct predictions. """
    correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip().lower() == ref.strip().lower())
    return correct / len(predictions) * 100 if predictions else 0

def analyze_errors(predictions, references):
    """ Analyzes and logs errors to help identify common issues or patterns in incorrect predictions. """
    error_details = []
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() != ref.strip().lower():
            error_details.append({'predicted': pred, 'reference': ref})
    print("Error Analysis: Errors found in predictions.")
    for error in error_details:
        print(f"Predicted: {error['predicted']} | Reference: {error['reference']}")

def main():
    api_key = '' # Enter your api key here
    data = load_data('preprocessed_data.csv')
    total_samples = len(data)
    predictions = []
    references = []

    for i, (input_text, correct_api) in enumerate(data):
        predicted_api = predict_with_gpt3(input_text, api_key)
        if predicted_api is not None:
            predictions.append(predicted_api)
            references.append(correct_api)
        else:
            predictions.append("")
            references.append(correct_api)

        # Update the loading bar
        progress = (i + 1) / total_samples
        bar_length = 50
        filled = int(progress * bar_length)
        bar = '█' * filled + '-' * (bar_length - filled)
        percentage = int(progress * 100)
        sys.stdout.write(f"\rProgress: [{bar}] {percentage}%")
        sys.stdout.flush()

    accuracy = calculate_accuracy(predictions, references)
    print(f"\nOverall Prediction Accuracy: {accuracy:.2f}%")  # Print accuracy after loading bar

    analyze_errors(predictions, references)

if __name__ == "__main__":
    main()