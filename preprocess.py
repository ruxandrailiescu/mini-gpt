import pandas as pd

def preprocess_data(csv_file):
    """ create a text file which contains all the lines of the dialogue with the player preprended """
    df = pd.read_csv(csv_file)
    df = df[(df['Player'].isna()==False) & (df['ActSceneLine'].to_string().strip()!="")]
    df['PrevPlayer'] = df['Player'].shift(1)
    mask = df['PrevPlayer'].isna() | (df['Player'] != df['PrevPlayer'])
    df.loc[mask, 'PlayerLine'] = '\n' + df['Player'] + ': ' + df['PlayerLine'].astype(str)
    with open('alllines_processed.txt', 'w') as f:
        for value in df['PlayerLine']:
            if str(value).startswith('\n'):
                f.write(str(value) + '\n')
            else:
                f.write('\t\t' + str(value) + '\n')