import pandas as pd

# Read the train and test data from the CSV files
train_df = pd.read_csv('train_unique_S0.csv')
test_df = pd.read_csv('test_unique_S0.csv')

# Extract the 'S0' columns from the dataframes
train_data = train_df['S0'].tolist()
test_data = test_df['S0'].tolist()

# Create a new list to store the combined data
combined_data = []

# Combine the train and test data
for i in range(len(train_data)):
    if i < len(test_data):
        combined_data.append([train_data[i], test_data[i]])
    else:
        combined_data.append([train_data[i], '-'])

# Convert the combined data into a DataFrame
combined_df = pd.DataFrame(combined_data, columns=['Train_S0', 'Test_S0'])

# Save the combined data to a new CSV file
combined_df.to_csv('combined_S0.csv', index=False)

print("Combined CSV file has been created successfully.")
