import pandas as pd
from transformers import BertTokenizer
import json
import csv
import re

# Convert JSON files into CSV
def json_to_csv(json_file_path, csv_filename):

    with open(json_file_path, 'r') as json_file:
        raw_data = json.load(json_file)

    fieldnames = ['emotion', 'sentence']

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        # Extract the first human speach from JSON and write data
        for example in raw_data:

            # Extract emotion labels
            profile_entry = example.get('profile', {})
            emotion_label = profile_entry.get('emotion').get('type', '')
           
            # Only retain the initial speech from human
            talk_entry = example.get('talk', {})
            content_value = talk_entry.get('content', {}).get('HS01', '')

            # Write data
            writer.writerow({
                'emotion': emotion_label,
                'sentence': content_value
            })

        print(f'CSV file "{csv_filename}" has been created.')


def process_data(file_path, bert_model_name='kykim/bert-kor-base'):
    """
    Function to preprocess and tokenize data using a specified BERT tokenizer.
    
    Parameters:
    - file_path (str): Path to the CSV file, containing 'emotion' and 'sentence' field.
    - bert_model_name (str): Name or path of the BERT model for tokenization (default is 'kykim/bert-kor-base').
    
    Returns:
    - processed_data (pd.DataFrame): Processed training data with additional columns for labels, tokenized sentences, and cleaned tokens.
    """
    # Load data
    processed_data = pd.read_csv(file_path, sep='\t')

    # Convert labels to 'pos' and 'neg'
    positive_emotions = ['E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69']
    processed_data['label'] = processed_data['emotion'].apply(lambda x: 1 if x in positive_emotions else 0)

    # Load and apply KoBERT tokenizer to sentence
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    processed_data['tokenized_sentence'] = processed_data['sentence'].apply(lambda x: tokenizer.tokenize(x))

    # Function to clean tokens
    def clean_tokens(tokens):
        cleaned_tokens = []
        for token in tokens:
            if token.startswith('##'):  # remove subwords
                continue
            token = re.sub(r'[^\w\s]', '', token)  # remove punctuation
            if token:  # if token is not empty
                cleaned_tokens.append(token)
        return cleaned_tokens

    # Apply clean_tokens to the tokenized_sentence columns in train_data and test_data
    processed_data['cleaned_tokens'] = processed_data['tokenized_sentence'].apply(clean_tokens)

    return processed_data
