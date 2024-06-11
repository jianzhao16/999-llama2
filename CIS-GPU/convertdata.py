import pandas as pd

# Load the Excel file
file_path = './PPD lit rev.xlsx'
df = pd.read_excel(file_path)

# Define columns to exclude
columns_to_exclude = ['Conclusions', 'S.no.']

# Combine all columns except 'Conclusions' and 'S.no.' into a new column 'Prompt'
columns_to_combine = [col for col in df.columns if col not in columns_to_exclude]
df['Prompt'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Rename 'Conclusions' to 'Completion'
df.rename(columns={'Conclusions': 'Completion'}, inplace=True)

# Select only the 'Prompt' and 'Completion' columns
df_combined = df[['Prompt', 'Completion']]

# Display the new dataframe to the user
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Combined Prompts and Completions", dataframe=df_combined)

# Display the first few rows of the new dataframe
df_combined.head()

# Save the resulting dataframe to a CSV file
output_path = 'ourdata-ppd.csv'
df_combined.to_csv(output_path, index=False)

output_path
