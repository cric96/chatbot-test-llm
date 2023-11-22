import pandas as pd
import argparse
# Input arguments
parser = argparse.ArgumentParser(description='Reduce the dataset to ')
parser.add_argument('--data-file', type=str, default='./data/sentences4.csv', help='input file path')
parser.add_argument('--store-file', type=str, default='./data/reduced.csv', help='input file path')
parser.add_argument('--portion', type=float, default=0.1, help='The portion of the dataset to be sampled')

args = parser.parse_args()

# Load your dataset
df = pd.read_csv(args.data_file)

# Column name that contains the class labels
class_column = 'Label'

# Calculate the distribution of the classes
class_distribution = df[class_column].value_counts(normalize=True)

# Determine the size of the sample you want to take (e.g., 20% of the original dataset)
sample_size = int(len(df) * args.portion)

# Sample the data
sampled_df = pd.DataFrame()  # Empty DataFrame to hold the sampled data

for class_value, proportion in class_distribution.items():
    n_samples = int(sample_size * proportion)
    class_sample = df[df[class_column] == class_value].sample(n_samples, replace=True)
    sampled_df = pd.concat([sampled_df, class_sample], axis=0)

# Reset index for the sampled DataFrame
sampled_df = sampled_df.reset_index(drop=True)

# Store the sampled data
sampled_df.to_csv(args.store_file, index=False)