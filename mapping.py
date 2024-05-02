import pandas as pd

# Load the data
df = pd.read_csv('twitter_combined.txt', sep=' ', header=None, names=['from', 'to'])

# Create a unique list of IDs
unique_ids = pd.unique(df[['from', 'to']].values.ravel('K'))

# Create a mapping of old IDs to new IDs
id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

# Apply mapping
df['from'] = df['from'].map(id_map)
df['to'] = df['to'].map(id_map)

# Save the mapped data to a new file
df.to_csv('mapped_twitter_combined.txt', sep=' ', header=False, index=False)

