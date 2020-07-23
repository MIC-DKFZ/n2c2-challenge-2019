import pandas as pd
import os
import re
import subprocess
from shutil import rmtree
from pathlib import Path
from mtc.challenge_pipelines.preprocess_data import generate_preprocessed_data

from mtc.settings import NLP_RAW_DATA, MEDEX_PATH


def check_dir_empty(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.makedirs(dir)


def generate_ingredients_file(folder, mode):
    # The java program must be executed in its the home directory
    current_dir = os.getcwd()
    os.chdir(MEDEX_PATH)

    # Make sure the folders exist and are empty
    input_dir = 'input'
    check_dir_empty(input_dir)
    output_dir = 'output'
    check_dir_empty(output_dir)

    # Appends ingredient information to an existing preprocessed file
    preprocess_file = Path(os.path.join(folder, f'preprocessed_data_{mode}.tsv'))
    df = pd.read_csv(preprocess_file, sep='\t')

    # Create a new file with one sentence at each line (required by the Java program
    df_sentences = pd.DataFrame(df['sentence a'].tolist() + df['sentence b'].tolist())
    df_sentences.to_csv(os.path.join(input_dir, preprocess_file.name), header=None, index=False)

    # Extract the ingredient information...
    if os.name == 'nt':
        delimiter = ';'
    else:
        delimiter = ':'

    subprocess.call(f'java -Xmx1024m -cp lib/*{delimiter}bin org.apache.medex.Main -i {input_dir} -o {output_dir} -b n -f y -p n -d y', shell=True)

    # ... and read the output
    df_tab = pd.read_csv(os.path.join(output_dir, preprocess_file.name), sep='\t|\\|', engine='python', header=None, names=['sen pos', 'sen', 'drug name', 'brand name', 'drug form', 'strength', 'dose amount', 'route', 'frequency', 'duration', 'necessity', 'UMLS CUI', 'RXNORM RxCUI', 'RXNORM RxCUI for generic name', 'generic name'])

    os.chdir(current_dir)

    def extract_ingredient(idx, sen_type):
        sen_pos = idx + 1
        if sen_type == 'b':
            sen_pos += len(df)

        rows = df_tab[df_tab['sen pos'] == sen_pos]

        # Not every sentence is a tablet sentence
        if rows.empty:
            return None

        # There should at least be e.g. the word tablet or an amount like 50 mg to count as tablet sentence
        if all(pd.isnull(rows['drug form'])) and all(pd.isnull(rows['strength'])):
            return None

        if any(~pd.isnull(rows['generic name'])):
            name = rows['generic name'].iloc[0]
        elif any(~pd.isnull(rows['drug name'])):
            name = rows['drug name'].iloc[0]
        else:
            return None

        name = name.lower()
        name = re.sub(r'\s*\([^)]+\)', '', name)
        name = re.sub(r',.*', '', name)
        name = name.split()[0]

        if name == 'multiple':
            name = 'vitamin'

        return name

    df['ingr_a'] = ''
    df['ingr_b'] = ''
    for i, row in df.iterrows():
        df.at[i, 'ingr_a'] = extract_ingredient(i, 'a')
        df.at[i, 'ingr_b'] = extract_ingredient(i, 'b')

    df.to_csv(os.path.join(preprocess_file.parent, preprocess_file.stem + '_with_ingredients.tsv'), sep='\t', index=False)


def get_ingredient(df_tab, idx, sen_type):
    ingr = df_tab.loc[idx, 'ingr_' + sen_type]
    if pd.isnull(ingr):
        return None
    else:
        return ingr


def add_ingredients(df_tab, df):
    df['ingr_a'] = ''
    df['ingr_b'] = ''

    for i, row in df.iterrows():
        df.at[i, 'ingr_a'] = get_ingredient(df_tab, df.loc[i, 'index'], 'a')
        df.at[i, 'ingr_b'] = get_ingredient(df_tab, df.loc[i, 'index'], 'b')


def ingredients(output_folder):
    folder_name = 'preprocessed_data_to_extract_ingredients'
    # Special preprocessing for easier extraction of the ingredients
    generate_preprocessed_data(['ContractionExpander', 'NumberUnifier', 'SpellingCorrector', 'MedicationRemover', 'SentenceTokenizer', 'WordTokenizer', 'PunctuationRemover', 'LowerCaseTransformer'], params=None, output_folder=output_folder,
                               folder_name=folder_name)

    generate_ingredients_file(output_folder / folder_name, 'train')
    generate_ingredients_file(output_folder / folder_name, 'test')


if __name__ == '__main__':
    base = 'preprocessed_data_2019-08-06_21-40-40'
    generate_ingredients_file(base, 'train')
    generate_ingredients_file(base, 'test')
