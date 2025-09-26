import argparse
import collections
import os, sys, logging
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import chromadb, gc

from . import constants

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


plt.style.use('classic')
sns.set_style('whitegrid', {'axes.grid': False})


############ LOGGING utils ############
def setup_logging():
    log_folder = os.path.join(constants.PROJ_DIR, 'logs')
    os.makedirs(log_folder, exist_ok=True)

    log_filename = f"runLog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_filepath = os.path.join(log_folder, log_filename)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Run log started...")


def get_logger(name):
    return logging.getLogger(name)


############ CLI utils ############
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM patient classification")
    parser.add_argument("--center", type=str, required=True,
                        help="Center (eg: 'UoM', 'SU')")
    parser.add_argument("--condition", type=int, required=True,
                        help="Ocular condition (eg: '1: glaucoma', '2: diabetic retinopathy', '3: age-related macular degeneration')")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (eg: 'llama3', 'mistral'). Please refer 'constants.py' for model names")
    parser.add_argument("--prompt_type", type=str, required=True,
                        help="Prompt type (eg: 'NULL_zeroShot', 'Prefix_zeroShot'). Please refer 'prompts.py' for prompt types")
    parser.add_argument("--is_include_clinical_reason", type=str2bool, required=True,
                        help="Do you want to exclude clinical reason response from LLM? (eg: 'True', 'False')")
    parser.add_argument("--note_selection_mode", type=str, choices=['one_note', 'multiple_notes', 'all_notes'],
                        required=True,
                        help="Criteria for selecting notes: 'one_note', 'multiple_notes', 'all_notes'")
    parser.add_argument("--n", type=int,
                        help="Number of recent notes for prompting (eg: '1', '4'). Required if criteria is 'multiple_notes'")
    parser.add_argument("--combine_notes", type=str2bool, default=False,
                        help="Flag to combine notes for 'multiple_notes' or 'all_notes' criteria (eg: 'True', 'False')")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for data loading (eg: '32', '64')")
    parser.add_argument("--max_patients_to_process", type=int, default=None,
                        help="Number of patients user want to process (eg: '100', '23'), If not specified, all patients will be processed.")

    args = parser.parse_args()

    if args.note_selection_mode == 'multiple_notes' and args.n is None:
        parser.error("--n is required when note_selection_mode is 'multiple notes'")
    return args


def initialize_parameters_for_condition(value):
    if value == 1:
        constants.CONDITION = 'glaucoma'
        constants.CONDITION_SPECIFIC_INSTRUCTION = constants.INSTRUCTION_FAMILY.get(str(value))
        constants.true_label_variable = 'GLA_DEPT_GRADE'
        print(f"CONDITION variable set to {constants.CONDITION}")
    elif value == 2:
        constants.CONDITION = 'diabetic retinopathy'
        constants.CONDITION_SPECIFIC_INSTRUCTION = constants.INSTRUCTION_FAMILY.get(str(value))
        constants.true_label_variable = 'DR_DEPT_GRADE'
        print(f"CONDITION variable set to {constants.CONDITION}")
    elif value == 3:
        constants.CONDITION = 'age-related macular degeneration'
        constants.CONDITION_SPECIFIC_INSTRUCTION = constants.INSTRUCTION_FAMILY.get(str(value))
        constants.true_label_variable = 'AMD_DEPT_GRADE'
        print(f"CONDITION variable set to {constants.CONDITION}")
    else:
        error_message = "Please check value for condition variable"
        raise ValueError(error_message)
    # print(constants.CONDITION_SPECIFIC_INSTRUCTIONS)


############ Data utils ############
def get_data(filename, folder_name):
    """
    Read a csv file from the 'data' folder which is in the parent directory of the script's directory.

    Parameters:
    filename (str): The name of the csv file (including the .csv extension)

    Returns:
    dataframe (pd.DataFrame): The contents of the csv file as a pandas DataFrame.
    """

    data_directory = os.path.join(constants.PROJ_DIR, folder_name)
    csv_path = os.path.join(data_directory, filename)

    # Check if the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"The file '{filename}' does not exist in the {data_directory} directory.")

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    return df


def get_recent_notes(center, criteria='one_note', n=None):
    center_constants = get_center_constants(center)
    base_df_ = get_data(center_constants["INPUT_FILE"], 'data')
    base_df = base_df_[['PAT_MRN']]
    base_df['PAT_MRN'] = base_df['PAT_MRN'].astype(str)
    # base_df.loc[:, 'PAT_MRN'] = base_df['PAT_MRN'].astype(str)

    notes_df = get_data(center_constants['ALL_NOTES'], 'data')
    notes_df['NOTE_DATE'] = pd.to_datetime(notes_df['NOTE_DATE'], format='%m/%d/%y')
    notes_df['PAT_MRN'] = notes_df['PAT_MRN'].astype(str)

    merged_df_ = pd.merge(base_df, notes_df, on='PAT_MRN', how='left')
    merged_df_1 = merged_df_.dropna()  # Drop rows with NA values
    merged_df = merged_df_1[~(merged_df_1 == '').any(axis=1)]
    sorted_notes_df = merged_df.sort_values(by=['PAT_MRN', 'NOTE_DATE'], ascending=[True, False])

    # for a given id, we have same date and enc. to resolve this we need to new variable to order
    sorted_notes_df['RANK'] = sorted_notes_df.groupby('PAT_MRN').cumcount(ascending=True)

    if criteria == 'one_note':
        # grouped_notes = sorted_notes_df.groupby('PAT_MRN').head(1)
        grouped_notes = sorted_notes_df[sorted_notes_df['RANK'] == 0]
    elif criteria == 'multiple_notes':
        if n is None:
            raise ValueError("Parameter n must be provided for criteria 'multiple_notes' ")
        # grouped_notes = sorted_notes_df.groupby('PAT_MRN').head(n)
        # grouped_notes = sorted_notes_df[sorted_notes_df['RANK'] < n]
        rslt_df = pd.DataFrame()
        for _, group in sorted_notes_df.groupby('PAT_MRN'):
            rslt_df = pd.concat([rslt_df, group.head(n) if n < len(group) else group])
        grouped_notes = rslt_df
    elif criteria == 'all_notes':
        grouped_notes = sorted_notes_df
    else:
        raise ValueError("Invalid criteria. Choose either 'one_note', 'multiple_notes', or 'all_notes'.")

    # concatenated_notes = grouped_notes.groupby('PAT_MRN')['PROGRESS_NOTE'].apply(
    #     lambda x: '\n*********************\n'.join([f"PATIENT NOTE:\n{nt}" for nt in x])).reset_index()
    # # lambda x: '\n*********************\n PATIENT NOTE \n'.join(x)).reset_index()
    # result_df = base_df.merge(concatenated_notes, on='PAT_MRN', how='left')

    # # Save the data to temp folder
    # temp_dir = os.path.join(constants.PROJ_DIR, 'data', 'temp')
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    #
    # print("Saving data file to temp folder")
    # result_df.to_csv(os.path.join(temp_dir, f'recent_notes_{n}_{constants.CONDITION}.csv'), index=False)

    # return grouped_notes[['PAT_MRN', 'PROGRESS_NOTE']]
    return grouped_notes[['PAT_MRN', 'PROGRESS_NOTE']].copy()


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pat_id = self.data.iloc[idx]['PAT_MRN']
        note = str(self.data.iloc[idx]['PROGRESS_NOTE'])
        return pat_id, note
        # return self.data.iloc[idx]['PAT_MRN'], self.data.iloc[idx]['PROGRESS_NOTE']


def prepare_data(center, criteria="one_note", n=None, batch_size=2, combine_notes=False):
    df = get_recent_notes(center, criteria, n)

    if combine_notes and criteria in ['multiple_notes', 'all_notes']:
        concatenated_notes = df.groupby('PAT_MRN')['PROGRESS_NOTE'].apply(
            lambda x: '\n*********************\n'.join([f"PATIENT NOTE:\n{nt}" for nt in x])).reset_index()
        concatenated_notes.columns = ['PAT_MRN', 'PROGRESS_NOTE']
        dataset = CustomDataset(concatenated_notes)
    else:
        dataset = CustomDataset(df)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # dataset.data.to_csv('a2.csv')
    return dataloader, dataset


############ PERFORMANCE utils ############

def get_accuracy(true_labels, predictions) -> float:
    correct_predictions = np.sum(true_labels == predictions)
    accuracy = correct_predictions / len(true_labels)
    return accuracy


def get_precision(true_labels, predictions) -> float:
    true_positive = np.sum((predictions == 1) & (true_labels == 1))
    false_positive = np.sum((predictions == 1) & (true_labels == 0))

    if true_positive + false_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    return precision


def get_recall(true_labels, predictions) -> float:
    true_positive = np.sum((predictions == 1) & (true_labels == 1))
    false_negative = np.sum((predictions == 0) & (true_labels == 1))

    if true_positive + false_negative == 0:
        return 0.0
    recall = true_positive / (true_positive + false_negative)
    return recall


def get_f1_score(true_labels, predictions) -> float:
    precision = get_precision(true_labels, predictions)
    recall = get_recall(true_labels, predictions)

    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_performance_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Tuple[
    Dict[str, float], Dict[str, Dict[str, float]]]:
    # accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
    # precision = sklearn.metrics.precision_score(true_labels, predictions, labels = [0,1])
    # recall = sklearn.metrics.recall_score(true_labels, predictions, labels = [0,1])
    # micro_f1 = sklearn.metrics.f1_score(true_labels, predictions, average='micro')
    # print(accuracy, precision, recall, micro_f1)

    accuracy = get_accuracy(true_labels, predictions)
    precision = get_precision(true_labels, predictions)
    recall = get_recall(true_labels, predictions)
    f1 = get_f1_score(true_labels, predictions)

    bootstraps = collections.defaultdict(list)
    np.random.seed(0)
    for i in range(2000):

        while True:
            bootstrap_index = np.random.randint(low=0, high=predictions.shape[0], size=predictions.shape[0])
            if np.all(bootstrap_index < predictions.shape[0]):
                break

        bootstraps['accuracy'].append(
            get_accuracy(true_labels[bootstrap_index], predictions[bootstrap_index]))
        bootstraps['precision'].append(
            get_precision(true_labels[bootstrap_index], predictions[bootstrap_index]))
        bootstraps['recall'].append(
            get_recall(true_labels[bootstrap_index], predictions[bootstrap_index]))
        bootstraps['f1'].append(
            get_f1_score(true_labels[bootstrap_index], predictions[bootstrap_index]))

    bootstraps = dict(bootstraps)
    CI_ranges = {}
    for key, vals in bootstraps.items():
        CI_ranges[key] = {
            'lower': np.percentile(vals, 2.5),
            'upper': np.percentile(vals, 97.5),
        }

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }, CI_ranges


def plot_confusion_matrix(cf_matrix, model_name, output_file, condition, grader_value):
    plt.figure(figsize=(7, 7))
    heatmap = sns.heatmap(cf_matrix, annot=True, fmt=".0f", cmap='Blues', annot_kws={"size": 20},
                          xticklabels=[f'No {condition}', condition],
                          yticklabels=[f'No {condition}', condition])
    plt.title(f"Confusion Matrix for {model_name}", fontsize=20)
    plt.ylabel('True Class', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=20)

    figures_dir = os.path.join(constants.PROJ_DIR, 'results', 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_filename = f"Confusion_Matrix_{output_file}_{grader_value}.png"
    fig_filepath = os.path.join(figures_dir, fig_filename)
    print(f"Saving {fig_filename} to {fig_filepath}")
    plt.savefig(fig_filepath, dpi=300)
    # plt.show()


def save_fp_plot(fp_counts, custom_tick_lables, center, grader_value):
    fp_df = pd.DataFrame(list(fp_counts.items()), columns=['Model', 'False Positive Counts'])
    plt.figure(figsize=(5, 5))
    plt.bar(fp_df['Model'], fp_df['False Positive Counts'], color="#FFA07A", width=0.4)
    plt.xlim(-0.5, len(fp_df['Model']) - 1 + 0.5)
    plt.title(f"False Positive Counts per Model")
    plt.ylabel('False Positive Count', fontsize=12)
    plt.xlabel('Large Language Model', fontsize=12)
    plt.xticks(ticks=range(len(fp_df)), labels=custom_tick_lables, rotation=0)

    figures_dir = os.path.join(constants.PROJ_DIR, 'results', 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_filename = f"FP_{center}_{grader_value}.png"
    fig_filepath = os.path.join(figures_dir, fig_filename)
    print(f"Saving {fig_filename} to {fig_filepath}")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_filepath, dpi=300)
    # plt.show()


############ extra utils ############

def get_n_value_for_note_selection_mode(criteria, n=None):
    if criteria == 'one_note':
        return 1
    elif criteria == 'multiple_notes':
        return n
    else:
        return 'all'


class Usage_stats:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def calculate_cost(model, usage_stats, dataset, file_name):
    completion_tokens = sum(stat.completion_tokens for stat in usage_stats)
    prompt_tokens = sum(stat.prompt_tokens for stat in usage_stats)

    cost_per_1k_completion_tokens: float = 0.0
    cost_per_1k_prompt_tokens: float = 0.0

    if model == 'gemma2':
        cost_per_1k_completion_tokens = 0.00015
        cost_per_1k_prompt_tokens = 0.00012
    elif model in ['llama3']:
        cost_per_1k_completion_tokens = 0.0002
        cost_per_1k_prompt_tokens = 0.00014
    elif model in ['llama31']:
        cost_per_1k_completion_tokens = 0.0002
        cost_per_1k_prompt_tokens = 0.00014
    elif model in ['mistral']:
        cost_per_1k_completion_tokens = 0.00025
        cost_per_1k_prompt_tokens = 0.00025
    elif model in ['med42']:  # 'qwen']:
        cost_per_1k_completion_tokens = 0.0002
        cost_per_1k_prompt_tokens = 0.00014
    elif model in ['yi']:
        cost_per_1k_completion_tokens = 0.00020
        cost_per_1k_prompt_tokens = 0.00015
    elif model in ['biomistral ']:  # 'qwen']:
        cost_per_1k_completion_tokens = 0.0002
        cost_per_1k_prompt_tokens = 0.00014
    elif model in ['qwen ']:
        cost_per_1k_completion_tokens = 0.00020
        cost_per_1k_prompt_tokens = 0.00015

    total_cost = ((completion_tokens * (cost_per_1k_completion_tokens / 1000)) +
                  (prompt_tokens * (cost_per_1k_prompt_tokens / 1000)))
    num_patients = dataset.data['PAT_MRN'].nunique()

    temp_dir = os.path.join(constants.PROJ_DIR, 'results', 'usage_costs')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    output_path = os.path.join(constants.PROJ_DIR, 'results', 'usage_costs', file_name + '_usage.txt')
    with open(output_path, 'w') as f:
        f.write(f"# of patients: {num_patients}\n")
        f.write(f"# of completion tokens: {completion_tokens}\n")
        f.write(f"# of prompt tokens: {prompt_tokens}\n")
        f.write(f"Total # of tokens: {completion_tokens + prompt_tokens}\n")
        f.write(f"Cost: ${total_cost:.4f}\n")

def get_token_usage(usage_stats):

    temp_dir = os.path.join(constants.PROJ_DIR, 'results', 'usage_costs')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    completion_tokens = sum(stat.completion_tokens for stat in usage_stats)
    prompt_tokens = sum(stat.prompt_tokens for stat in usage_stats)
    total_tokens = completion_tokens + prompt_tokens

    return prompt_tokens, completion_tokens, total_tokens


def get_center_constants(center):
    if center not in constants.CENTER_CONSTANTS:
        raise ValueError(f"Center '{center}' is not defined in constants.py")
    return constants.CENTER_CONSTANTS[center]

########### RAG utils ############

def initialize_chromadb(collection_name: str, embed_model: str, db_collection: bool = False):
    """
    Initializes and returns a ChromaDB client and collection.

    Args:
        collection_name: The name of the ChromaDB collection.
        embed_model: The name of the embedding model (used for naming the collection).
        db_collection: Whether to delete the collection if it already exists.

    Returns:
        A tuple containing the ChromaDB client and the ChromaDB collection.
    """
    os.makedirs(constants.CHROMA_DB_PATH, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=chromadb_path)

    if db_collection:
        try:
            existing_collection = chroma_client.get_collection(name=collection_name)
            if existing_collection is not None:
                print(f"Deleting existing collection: '{collection_name}'...")
                chroma_client.delete_collection(collection_name)
            else:
                print(f"Collection '{collection_name}' does not exist, skipping deletion.")
        except ValueError as e:
            print(f"Collection1 '{collection_name}' does not exist, skipping deletion.")

    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return chroma_client, chroma_collection
