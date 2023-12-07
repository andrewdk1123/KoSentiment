import os
import csv

RAW_DATA_DIR = './data/raw-data/'
AUGMENTED_DATA_DIR = './data/data-augmentor/'
DELIMITER = '\t'

def merge_csv_files(output_file_path, training_file='training.csv'):
    combined_data = []
    header = ['emotion', 'sentence'] 

    # Read train.csv
    raw_data = os.path.join(RAW_DATA_DIR, training_file)
    with open(raw_data, 'r', newline='') as data:
        train_reader = csv.reader(data, delimiter=DELIMITER)
        header = next(train_reader, header)
        combined_data.extend(train_reader)

    # Read augmented data
    for i in range(1, 11):
        file_to_combine = os.path.join(AUGMENTED_DATA_DIR, f'augmented_data_chunk{i}.csv')

        with open(file_to_combine, 'r', newline='') as input_file:
            reader = csv.reader(input_file, delimiter=DELIMITER)
            if not header:
                header = next(reader, header)  
            combined_data.extend(reader)

    # Create a CSV writer for the output file
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=DELIMITER)
        writer.writerow(header)
        writer.writerows(combined_data)

def main():
    output_filename = './data/combined_train_data.csv'
    merge_csv_files(output_filename)

if __name__ == '__main__':
    main()
