import os
from src.utils import basic_utils
import src.utils.constants as constants


def sanity_check():

    errors_df_temp_dir = os.path.join(constants.PROJ_DIR, 'results', 'invalid_outputs')
    if not os.path.exists(errors_df_temp_dir):
        os.makedirs(errors_df_temp_dir)

    results_directory = os.path.join(constants.PROJ_DIR, 'results')
    output_files = [f for f in os.listdir(results_directory) if f.endswith('.csv')]

    for output_file in output_files:
        print(output_file)
        prediction_df = basic_utils.get_data(output_file, 'results')

        empty_rows = prediction_df[prediction_df.isnull().any(axis=1) |
                                   (prediction_df.eq("").any(axis=1))]
        print(empty_rows)

        invalid_entries = prediction_df[~prediction_df['Condition_Status'].astype(str).isin(['0','1']) |
                                        prediction_df['Condition_Status'].isnull() |
                                        (prediction_df['Condition_Status'] == " ")]

        if not invalid_entries.empty:
            print(f"File {output_file} has {len(invalid_entries)} invalid outputs")
            invalid_entries_df = invalid_entries[['PAT_MRN', 'Condition_Status']]
            filename = output_file.replace('.csv', '_invalid_entries.csv')
            filepath = os.path.join(errors_df_temp_dir, filename)
            invalid_entries_df.to_csv(filepath, index=False)
        else:
            print("No invalid outputs")


def main():
    sanity_check()


if __name__ == "__main__":
    main()