from transformers import BertTokenizer
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import re
import os

AUGMENTED_DIR = './augmented_data/'

def load_json(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f'Error: The specified JSON file "{file_path}" does not exist.')
        return None

def save_csv(data, csv_filename, fieldnames):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)

# def clean_tokens(sentence):
#     # Function to clean tokens
#     cleaned_tokens = []

#     for token in sentence:
#         # Skip tokens starting with '#' or '<'
#         if token.startswith('#') or token.startswith('<'):
#             continue

#         # Remove non-alphanumeric characters and non-Korean characters using regex
#         token = re.sub(r'[^a-zA-Z0-9\uAC00-\uD7A3\s]', '', token)

#         # If the token is not empty after cleaning, append to the list
#         if token:
#             cleaned_tokens.append(token)

#     return cleaned_tokens

def process_data(file_path):
    
#    tokenizer = BertTokenizer.from_pretrained(bert_model_name)     
    
    if file_path.startswith(AUGMENTED_DIR):

        df = pd.read_csv(file_path)

        # Convert labels to 'pos' and 'neg'
        positive_emotions = ['E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69']
        df['label'] = df['emotion'].apply(lambda x: 1 if x in positive_emotions else 0)

        df['generated_sentence'] = df['generated_sentence'].astype(str)
        df['sentence'] = df['generated_sentence'].apply(lambda x: re.sub(r'\n(?=(?:(?:[^"]*"){2})*[^"]*$)', ' ', x))            
        df = df.drop('generated_sentence', axis=1)

        # Tokenize and clean sentences
        #df['tokenized_sentence'] = df['sentence'].apply(lambda x: tokenizer.tokenize(x))
        #df['cleaned_tokens'] = df['tokenized_sentence'].apply(clean_tokens)

        return df

    else:

        df = pd.read_csv(file_path, sep='\t')

        # Convert labels to 'pos' and 'neg'
        positive_emotions = ['E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69']
        df['label'] = df['emotion'].apply(lambda x: 1 if x in positive_emotions else 0)

        # Tokenize and clean sentences
        #df['tokenized_sentence'] = df['sentence'].apply(lambda x: tokenizer.tokenize(x))
        #df['cleaned_tokens'] = df['tokenized_sentence'].apply(clean_tokens)

        return df

def main():
    if not os.listdir(AUGMENTED_DIR):
        print("Processing raw data.")

        # JSON to CSV
        raw_training_data = load_json('training_raw.json')
        raw_test_data = load_json('test_raw.json')

        save_csv(raw_training_data, 'training.csv', ['emotion', 'sentence'])
        save_csv(raw_test_data, 'test.csv', ['emotion', 'sentence'])

        # Process data
        training_data = process_data('training.csv')
        y = training_data['label']
        counts = y.value_counts()

        counts.plot(kind='pie', labels=['neg', 'pos'], autopct='%1.1f%%')
        plt.savefig('train_label_dist.png')
        print('"train_label_dist.png" has been created.')

        pos_training_data = training_data[training_data['label'] == 1]
        pos_training_data.to_csv(os.path.join(AUGMENTED_DIR, 'pos_training.csv'))
        print(f"'{os.path.join(AUGMENTED_DIR, 'pos_training.csv')}' has been created.\nLet's open `KoGPT-2 Data Augmentor.ipynb` and augment positive examples!")
    else:
        # Process training and augmented training data
        training_data = process_data('training.csv')
        
        for i in range(1, 11):
            filename = os.path.join(AUGMENTED_DIR, f'augmented_df ({i}).csv')
            print(f'processing {filename}...')
            processed_df = process_data(filename)

            # Append the processed DataFrame to the training_data
            training_data = pd.concat([training_data, processed_df], ignore_index=True)

        # Perform random undersampling  
        y = training_data['label']
        X = training_data.drop('label', axis=1)

        # Create a RandomUnderSampler instance
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        undersampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        undersampled_df['label'] = y_resampled

        counts = y_resampled.value_counts()

        counts.plot(kind='pie', labels=['neg', 'pos'], autopct='%1.1f%%')
        plt.savefig('augmented_train_label_dist.png')

        print(y_resampled.value_counts())

        undersampled_df.to_csv('./data/processed_training.csv')

        # Process test data
        test_data = process_data('test.csv')
        test_data.to_csv('./data/processed_test.csv')

if __name__ == "__main__":
    main()
