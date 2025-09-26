import os.path, itertools
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

import utils.basic_utils as basic_utils
from utils.constants import PROJ_DIR
import utils.constants as constants

def analyze_row_difference(true_labels, predictions):

    missing_ids = true_labels[~true_labels['PAT_MRN'].isin(predictions['PAT_MRN'])]
    print(f'Total number of ids LLM did not process: {missing_ids.shape[0]}')

    original_value_counts = true_labels['Binary_true_label'].value_counts()
    print(f'Count of labels in original dataset: {original_value_counts}')

    if missing_ids.shape[0] > 0:
        label_counts = missing_ids['Binary_true_label'].value_counts()
        print(f'Count of missing labels in predictions:')
        print(label_counts)
        print("###########################")

def save_results_to_files(df, analysis_type, center, grader_value, phe_type):

    temp_dir = os.path.join(constants.PROJ_DIR, 'results', analysis_type)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    output_filename = f"{analysis_type}_{phe_type}_{grader_value}_{center}_ICD.csv"
    output_filepath = os.path.join(temp_dir, output_filename)

    df.to_csv(output_filepath, float_format='%.2f', index=False)
    print(f"{analysis_type} results saved to {output_filepath}")


def evaluate_performance(center, grader_value, phe_constants, initialize_parameters_for_condition_value):

    center_constants = basic_utils.get_center_constants(center)
    basic_utils.initialize_parameters_for_condition(initialize_parameters_for_condition_value)

    true_label_df = basic_utils.get_data(center_constants["INPUT_FILE"], 'data')
    NA_count = true_label_df[grader_value].isnull().sum()
    print(f"Total number of NAs: {NA_count} ") # true_label_df[grader_value].value_counts(dropna=False))
    if NA_count > 0:
        true_label_df = true_label_df.dropna(subset=[grader_value])
        # print(true_label_df[grader_value].value_counts(dropna=False))
        print("NAs dropped")

    # true_label_df['Binary_true_label'] = true_label_df[constants.true_label_variable].map({"NO EVIDENCE": 0,
    #                                                                                    "DEFINITE": 1})
    true_label_df['Binary_true_label'] = true_label_df[grader_value].map({"NO EVIDENCE": 0,
                                                                                           "DEFINITE": 1})
    print(f"Condition: {constants.CONDITION}")
    true_labels = true_label_df[['PAT_MRN', 'Binary_true_label']]

    results_directory = os.path.join(PROJ_DIR, 'data')
    output_files = [f for f in os.listdir(results_directory) if f.startswith(phe_constants["INPUT_FILE"])]

    all_metrics_df = pd.DataFrame()
    all_confusion_matrix = pd.DataFrame()
    all_classification_report = pd.DataFrame()

    for output_file in output_files:

        print(f'\nPerformance metrics for {output_file}:')
        prediction_df = basic_utils.get_data(output_file, 'data')
        # predictions = prediction_df
        predictions = prediction_df[['PAT_MRN', phe_constants["LABEL_COLUMN"]]]

        analyze_row_difference(true_labels, predictions)
        merged_df = true_labels.merge(predictions, on = 'PAT_MRN', how = "left")

        if merged_df.isnull().values.any():
            print("NaN values present\n", merged_df.isnull().sum())
            merged_df = merged_df.dropna(subset=['Binary_true_label',phe_constants["LABEL_COLUMN"]])
            print("NaN values present\n", merged_df.isnull().sum())
        print(list(merged_df.columns))

        # metrics, ci_range = basic_utils.calculate_performance_metrics(merged_df[constants.true_label_variable], merged_df['Condition_Status'])
        metrics, ci_range = basic_utils.calculate_performance_metrics(merged_df['Binary_true_label'].values,
                                                                      merged_df[phe_constants["LABEL_COLUMN"]].values)
        all_metrics = None
        all_metrics = {'filename': output_file}
        for metric, value in metrics.items():
            lower_ci = ci_range[metric]['lower']
            higher_ci = ci_range[metric]['upper']
            print(f"{metric}: {value:.2f} [{lower_ci:.2f} - {higher_ci:.2f}]")
            all_metrics[metric] = f"{value:.2f} [{lower_ci:.2f} - {higher_ci:.2f}]"

        all_metrics_ = pd.DataFrame([all_metrics])
        all_metrics_df = pd.concat([all_metrics_df, all_metrics_], ignore_index=True)
        # print(all_metrics_df)

        # Error Analysis Files
        # error_analysis(merged_df, output_file, grader_value)

        # Confusion matrix
        cf_matrix = confusion_matrix(merged_df['Binary_true_label'], merged_df[phe_constants["LABEL_COLUMN"]])
        basic_utils.plot_confusion_matrix(cf_matrix, output_file.split('_')[1].capitalize(), output_file,
                                          constants.CONDITION.capitalize(), grader_value)

        cnf_matrix_df = pd.DataFrame(cf_matrix,
                                     index=['True_Negative', 'True_Positive'],
                                     columns=['Predicted_Negative', 'Predicted_Positive'])
        cnf_matrix_df['filename'] = output_file
        cnf_matrix_df = cnf_matrix_df.reset_index().melt(id_vars=['index', 'filename'], var_name='Predicted', value_name='Count')
        cnf_matrix_df = cnf_matrix_df.rename(columns={'index': 'Actual'})
        all_confusion_matrix = pd.concat([all_confusion_matrix, cnf_matrix_df], ignore_index=True)


        # Classification report
        clf_report = classification_report(merged_df['Binary_true_label'], merged_df[phe_constants["LABEL_COLUMN"]], output_dict= True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df['filename'] = output_file
        clf_report_df = clf_report_df.reset_index().rename(columns={'index': 'Class'})
        all_classification_report = pd.concat([all_classification_report, clf_report_df], ignore_index=True)

    save_results_to_files(all_metrics_df, 'performance_analysis', center, grader_value, phe_constants['OUTPUT_FILE_NAME'])
    save_results_to_files(all_confusion_matrix, 'confusion_matrix', center, grader_value, phe_constants['OUTPUT_FILE_NAME'])
    save_results_to_files(all_classification_report, 'classification_report', center, grader_value, phe_constants['OUTPUT_FILE_NAME'])


def main():

    PHENOTYPE_CONSTANTS = {
        "GLA_SU": {
            "LABEL_COLUMN": "GLA_icd_bfr",
            "INPUT_FILE": "SU_ICD.csv",
            "OUTPUT_FILE_NAME": "GLA_ICD",
        },
        "GLA_UoM": {
            "LABEL_COLUMN": "GLA_icd_bfr",
            "INPUT_FILE": "UoM_ICD.csv",
            "OUTPUT_FILE_NAME": "GLA_ICD",
        },
        "DR_SU": {
            "LABEL_COLUMN": "DR_ICD_bfr",
            "INPUT_FILE": "SU_ICD.csv",
            "OUTPUT_FILE_NAME": "DR_ICD",
        },
        "DR_UoM": {
            "LABEL_COLUMN": "DR_ICD_bfr",
            "INPUT_FILE": "UoM_ICD.csv",
            "OUTPUT_FILE_NAME": "DR_ICD",
        },
        "AMD_SU": {
            "LABEL_COLUMN": "amdicd_bfr",
            "INPUT_FILE": "SU_ICD.csv",
            "OUTPUT_FILE_NAME": "AMD_ICD",
        },
        "AMD_UoM": {
            "LABEL_COLUMN": "amdicd_bfr",
            "INPUT_FILE": "UoM_ICD.csv",
            "OUTPUT_FILE_NAME": "AMD_ICD",
        },
    }

    # center = "UoM"
    # phe_type = "GLA"
    # grader_value = "GLA_DEPT_GRADE"
    # evaluate_performance(center, grader_value, PHENOTYPE_CONSTANTS[f'{phe_type}_{center}'])
    # print(PHENOTYPE_CONSTANTS[f'{phe_type}_{center}'])
    centers = ["SU"]
    phe_type = ["AMD"]
    # graders = ["GLA_GR1", "GLA_GR2", "GLA_DEPT_GRADE"]
    graders = ["AMD_DEPT_GRADE"]
    initialize_parameters_for_condition_value = 3

    for center, phe_type, grader in itertools.product(centers, phe_type, graders):
        key = f"{phe_type}_{center}"
        if key not in PHENOTYPE_CONSTANTS:
            print(f"PHENOTYPE_CONSTANTS missing for the key: {key}. Skipping this key")
            continue

        phe_constants = PHENOTYPE_CONSTANTS[key]
        print(f"Getting performance metrics for Center: {center}, Phenotype: {phe_type}, Grader: {grader}")
        evaluate_performance(center, grader, phe_constants, initialize_parameters_for_condition_value)

if __name__ == "__main__":
    main()
