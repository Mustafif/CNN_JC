import pandas as pd

# Function to read the CSV file and extract unique values of S0
def extract_unique_S0(file_path, output_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Extract unique values of S0
    unique_S0 = df['S0'].unique()

    # Save the unique S0 values to a new CSV file
    unique_S0_df = pd.DataFrame(unique_S0, columns=['S0'])
    unique_S0_df.to_csv(output_file_path, index=False)

    print(f"Unique S0 values from {file_path} saved to {output_file_path}")

# File paths to the train and test datasets
train_file_path = 'train_dataset.csv'
test_file_path = 'test_dataset.csv'

# Output file paths for the unique S0 values
train_unique_S0_file_path = 'train_unique_S0.csv'
test_unique_S0_file_path = 'test_unique_S0.csv'

# Extract and save unique S0 values from the train dataset
extract_unique_S0(train_file_path, train_unique_S0_file_path)

# Extract and save unique S0 values from the test dataset
extract_unique_S0(test_file_path, test_unique_S0_file_path)
