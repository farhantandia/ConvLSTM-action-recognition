import pandas as pd
df = pd.read_csv('data_file.csv')
df.to_csv('data_file1.csv', index=False)