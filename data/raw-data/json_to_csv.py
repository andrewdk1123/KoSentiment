import json
import csv

POSITIVE_EMOTION = {'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69'}
RAW_TRAIN_JSON = './data/raw-data/training_raw.json'
RAW_TEST_JSON = './data/raw-data/test_raw.json'
TRAIN_CSV = './data/raw-data/training.csv'
TEST_CSV = './data/raw-data/test.csv'

def json_to_csv(json_filename, csv_filename):
    with open(json_filename, 'r') as json_file:
        raw_data = json.load(json_file)
    
    fieldnames = ['emotion', 'sentence']

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        
        # Create CSV file with tab delimiter
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter='\t')
        csv_writer.writeheader()

        # Extract profile.emotion.type and talk.content.HS01 for emotion and sentence
        for example in raw_data:

            emotion = example.get('profile', {}).get('emotion', {}).get('type', '')
            sentence = example.get('talk', {}).get('content', {}).get('HS01', '')

            # Convert emotion to 1 if positive, else 0
            emotion = 1 if emotion in POSITIVE_EMOTION else 0

            # Write data to the created CSV file
            csv_writer.writerow({
                'emotion': emotion,
                'sentence': sentence
            })

        print(f'CSV file "{csv_filename}" has been created.')

def main():
    json_to_csv(RAW_TRAIN_JSON, TRAIN_CSV)
    json_to_csv(RAW_TEST_JSON, TEST_CSV)

if __name__ == '__main__':
    main()