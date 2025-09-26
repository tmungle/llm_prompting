import json, os, re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoTokenizer, AutoModel
from outlines.integrations.transformers import JSONPrefixAllowedTokens
from tqdm import tqdm
import torch
import time, datetime, shutil

from . import basic_utils, constants, schemas
# from src.utils import basic_utils, constants, schemas

def initialize_local_model(model_name):
    model_str = model_name.replace('/', '-')
    model_folder_name = f"model-{model_str}"
    path_to_model = os.path.join(constants.MODEL_DIR_PATH, model_folder_name)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path_to_model, local_files_only=True,
                                                 # torch_dtype="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=constants.DEVICE)
    print(next(model.parameters()).dtype)
    return tokenizer, model

def initialize_local_embed_model(model_name):
    model_str = model_name.replace('/', '-')
    model_folder_name = f"model-{model_str}"
    path_to_model = os.path.join(constants.MODEL_DIR_PATH, model_folder_name)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model, local_files_only=True)
    model = AutoModel.from_pretrained(path_to_model, local_files_only=True,
                                                 # torch_dtype="auto",
                                                 torch_dtype=torch.float32, # torch.bfloat16,
                                                 device_map=constants.DEVICE)

    print(next(model.parameters()).dtype)
    return tokenizer, model

def initialize_online_model(model_id):
    """
    This function initializes a model and tokenizer from pretrained versions using the specified names.
    Args:
    tokenizer_name: str, name of the tokenizer
    model_name: str, name of the model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=constants.HF_TOKEN, cache_dir=constants.CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=constants.HF_TOKEN, cache_dir=constants.CACHE_DIR,
                                                 device_map=constants.DEVICE)

    return tokenizer, model

def get_model_context_length(model):
    print(model.config.max_position_embeddings)


def calculate_total_tokens(dataset, tokenizer):

    total_tokens = 0

    for _, note in dataset:
        tokens = tokenizer.encode(note, add_special_tokens = False)
        total_tokens += len(tokens) + 64 # 64 are extra tokens just to be on safe side

    return total_tokens

def truncate_notes(tokenizer, note, max_tokens):
    encoded_tokens = tokenizer.encode(note)
    was_truncated = len(encoded_tokens) > max_tokens
    truncated_tokens = encoded_tokens[:max_tokens] if was_truncated else encoded_tokens
    truncated_note = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    return truncated_note, was_truncated


def process_llm_response(output):
    status = '-999'
    clinical_reason = 'NULL'

    processed_output = str(output[0])[1:-1]
    start = "generated_text': "
    # start = "Please provide your response:"
    start_index = processed_output.find(start)
    # print("p1")
    if start_index != -1:
        jsn_text = processed_output[start_index + len(start):].lower()
    else:
        return 'Start delimiter not found in text.'

    extracted_text = re.findall(r'\{.*?\}', jsn_text, re.DOTALL)
    if not extracted_text:
        extracted_text = re.findall(r'\{[^}]*', jsn_text, re.DOTALL)

    # Verify if we have found any JSON objects
    if not extracted_text:
        return 'No valid JSON found.'

    try:
        json_str = extracted_text[0]
        data = json.loads((json_str.encode().decode('unicode-escape')))
        for key in data:
            if 'status' in key.lower():
                status = data[key]
            if 'clinical_reason' in key.lower():
                clinical_reason = data[key]
    except json.JSONDecodeError:  # Handle the JSON decoding error
        return json_str, 'NULL'
    # except Exception as e:
    #     print(e)

    return status, clinical_reason


def run_llm_prompt(pipe, prefix_allowed_tokens_fn, task_with_note, is_include_clinical_reason):
    retries = 3
    attempt = 1
    response = "NULL"
    condition_status = -999
    clinical_reason = ''
    logger = basic_utils.get_logger(__name__)

    for attempt in range(0, retries):
        try:
            response = pipe(
                task_with_note,
                return_full_text=False, # If you want to set it True, please modify 'start' variable in process_llm_response()
                max_new_tokens=512, do_sample=None,
                temperature=0.0, top_p=0.9,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            # print("task_with_note")
            # condition_status = process_response(model_name, response)
            condition_status, clinical_reason = process_llm_response(response)
            # print(condition_status, clinical_reason)
            # print("!!!!")

            if condition_status in [0, 1]:
                if is_include_clinical_reason:
                    return response, condition_status, clinical_reason
                else:
                    clinical_reason = "NONE"
                    return response, condition_status, clinical_reason
        except Exception as e:
            logger.info(f'Attempt {attempt+1} failed with error {e}')

        time.sleep(2)



    return response, condition_status, clinical_reason


def setup_pipeline(model_name, is_include_clinical_reason, prompt_type):
    logger = basic_utils.get_logger(__name__)
    tokenizer, model = initialize_local_model(constants.MODEL_FAMILY.get(model_name))
    logger.info("Model and tokenizer initialized")

    current_schema = schemas.status_n_reason_schema if is_include_clinical_reason else schemas.status_only_schema

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prefix_allowed_tokens_fn = JSONPrefixAllowedTokens(
        schema=current_schema, tokenizer_or_pipe=pipe, whitespace_pattern=r" ?"
    )

    # Get the instructions
    prompt_type_ = constants.PROMPT_FAMILY.get(prompt_type)
    prompt_instructions_ = prompt_type_(is_include_clinical_reason)
    prompt_instructions__ = prompt_instructions_.replace("{condition}", constants.CONDITION)
    prompt_instructions = prompt_instructions__.replace("{condition_specific_instructions}",
                                                        constants.CONDITION_SPECIFIC_INSTRUCTION)

    # prompt_tokens = len(tokenizer.encode(prompt_instructions))
    # print(prompt_instructions)
    return tokenizer, pipe, prefix_allowed_tokens_fn, prompt_instructions


# Below function can be used for note level prompting
def execute_pipeline(center, dataset, condition, model_name, prompt_type, is_include_clinical_reason, criteria, n, max_patients_to_process=None):
    basic_utils.initialize_parameters_for_condition(condition)
    logger = basic_utils.get_logger(__name__)
    tokenizer, pipe, prefix_allowed_tokens_fn, prompt_instructions = setup_pipeline(model_name,
                                                                                    is_include_clinical_reason,
                                                                                    prompt_type)
    logger.info("Pipeline initialized")

    # Create a temporary directory and files for intermediate results within the results path
    result_filename = f"{constants.CONDITION}_{model_name}_{prompt_type}_{is_include_clinical_reason}_{criteria}_{n}_{center}"
    temp_dir = os.path.join(constants.PROJ_DIR, 'results', f'{result_filename}_temp_results')
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist
    temp_dir_usage = os.path.join(constants.PROJ_DIR, 'results', f'{result_filename}_temp_results_usage')
    os.makedirs(temp_dir_usage, exist_ok=True)
    temp_output_path = os.path.join(temp_dir, result_filename + '.csv')
    temp_output_path_usage = os.path.join(temp_dir_usage, result_filename + '_usage.csv')

    # Get patients to process
    processed_patients = set()
    for temp_file in os.listdir(temp_dir):
        if temp_file.endswith('.csv') and result_filename in temp_file:
            temp_df = pd.read_csv(os.path.join(temp_dir, temp_file))
            # temp_df = basic_utils.get_data(temp_file, 'result/temp_results')
            processed_patients.update(temp_df['PAT_MRN'].unique())
            # print(list(processed_patients))

    processed_patients_list = [str(pat_id) for pat_id in processed_patients]
    all_unique_patients = dataset.data['PAT_MRN'].unique()
    unique_patients = [pat_id for pat_id in all_unique_patients if str(pat_id) not in list(processed_patients_list)]
    unique_patients_to_process = unique_patients
    if max_patients_to_process is not None:
        unique_patients_to_process = unique_patients[:max_patients_to_process]

    max_patient_results, usage_stats, usage_df = [], [], []

    # Process max patient (each patient at a time)
    for patient_id in tqdm(unique_patients_to_process, desc="Processing patients"):
        if patient_id in processed_patients:
            logger.info(f"Skipping already processed patient id: {patient_id}")
            continue  # Skip already processed patients

        patient_notes = dataset.data[dataset.data['PAT_MRN'] == patient_id]['PROGRESS_NOTE'].tolist()
        individual_pat_results = []
        number_of_notes_skipped = 0

        for note in patient_notes:
            logger.info(f"***** {patient_id} *****")

            prompt_tokens = len(tokenizer.encode(prompt_instructions))
            truncated_note, was_truncated = truncate_notes(tokenizer, note,
                                                           max_tokens=constants.MODEL_MAX_TOKENS[
                                                                          model_name] - prompt_tokens - 64)

            task_with_note = prompt_instructions.replace("{note}", truncated_note)
            # print(task_with_note)
            try :
                # generated_response, condition_status, clinical_reason = run_llm_prompt(pipe, prefix_allowed_tokens_fn,
                #                                                                            task_with_note,
                #                                                                            is_include_clinical_reason)
                generated_response = "hi"
                condition_status = 1
                clinical_reason = "w"
                if condition_status not in [0,1]:
                    logger.info(f"Skipping the note for {patient_id} as condition is not 0 or 1")
                    number_of_notes_skipped += 1
                    continue

            except Exception as e:
                logger.info(f"Error in processing note for patient id {patient_id}: {e}")
                number_of_notes_skipped += 1
                continue
            # print(condition_status)
            response_tokens = 1 #len(tokenizer.encode(generated_response[0]['generated_text']))
            usage_stats.append(basic_utils.Usage_stats(prompt_tokens, response_tokens))

            individual_pat_results.append({'PAT_MRN': patient_id,
                                  'Task': task_with_note,
                                  'Generated_Text': json.dumps(generated_response[0]).encode().decode('unicode-escape'),
                                  'Condition_Status': condition_status,
                                  'Clinical_Reason': clinical_reason,
                                  'Truncation_Status': was_truncated
                                  })

        # After processing all notes for the patient, save the results if any were generated
        total_number_of_notes = len(patient_notes)
        if number_of_notes_skipped == total_number_of_notes and not individual_pat_results:
            individual_pat_results.append({
                'PAT_MRN': patient_id,
                'Task': None,
                'Generated_Text': None,
                'Condition_Status': 0,
                'Clinical_Reason': None,
                'Truncation_Status': None
            })
        max_patient_results.extend(individual_pat_results)

    # Get total tokens
    prompt_tokens, completion_tokens, total_tokens = basic_utils.get_token_usage(usage_stats)
    usage_df.append({
        'Prompt_Tokens': prompt_tokens,
        'Completion_Tokens': completion_tokens,
        'Total_Tokens': total_tokens
    })
    # After processing max patients, save the results to a temporary CSV file
    if max_patient_results:
        max_patient_results = pd.DataFrame(max_patient_results)
        if os.path.exists(temp_output_path):
            max_patient_results.to_csv(temp_output_path, mode='a', header=False, index=False)
            pd.DataFrame(usage_df).to_csv(temp_output_path_usage, mode='a', header=False, index=False)
            logger.info(f"Appended results to existing file: {temp_output_path} and {temp_output_path_usage}")
        else:
            max_patient_results.to_csv(temp_output_path, index=False)
            pd.DataFrame(usage_df).to_csv(temp_output_path_usage, index=False)
            logger.info(f"All intermediate results saved to {temp_output_path} and {temp_output_path_usage}")
    else:
        logger.info("No valid results to save.")

    # Check if the number of unique patient IDs in the temporary file matches the dataset to create final dataset
    temp_results_df = pd.read_csv(temp_output_path)
    temp_results_usage_df = pd.read_csv(temp_output_path_usage)
    # temp_results_df = basic_utils.get_data(f"{result_filename}.csv", 'result/temp_results')
    if len(temp_results_df['PAT_MRN'].unique()) == len(all_unique_patients):
        logger.info("Ids in temp file and dataset are same")
        grouped_df = temp_results_df.groupby('PAT_MRN')
        final_results = []

        for patient_id, group in tqdm(grouped_df, desc="Selecting notes with status"):
            final_status_per_patient = group.loc[group['Condition_Status'].idxmax()]
            final_results.append({
                'PAT_MRN': patient_id,
                'Task': final_status_per_patient['Task'],
                'Generated_Text': final_status_per_patient['Generated_Text'],
                'Condition_Status': final_status_per_patient['Condition_Status'],
                'Clinical_Reason': final_status_per_patient['Clinical_Reason'],
                'Truncation_Status': final_status_per_patient['Truncation_Status']
            })

        usage_sum = temp_results_usage_df.sum().to_frame().T

        # Save the final results to a new CSV file
        final_results_df = pd.DataFrame(final_results)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_result_output_path = os.path.join(constants.PROJ_DIR, 'results', f"{result_filename}_{timestamp}.csv")
        final_results_df.to_csv(final_result_output_path, index=False)
        logger.info(f"Final results saved to {final_result_output_path}")

        final_all_result_output_path = os.path.join(constants.PROJ_DIR, 'results', f"{result_filename}_{timestamp}_allNotes.csv")
        temp_results_df.to_csv(final_all_result_output_path, index=False)
        logger.info(f"Final results with all notes saved to {final_result_output_path}")

        usage_cost_path = os.path.join(constants.PROJ_DIR, 'results','usage_costs', f"{result_filename}_{timestamp}_usage_costs.csv")
        pd.DataFrame(usage_sum).to_csv(usage_cost_path, index=False)
        logger.info(f"Usage results for all notes saved to {usage_cost_path}")

        # Delete the temporary directory after successful processing
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_dir_usage)
        logger.info(f"Temporary directories '{temp_dir}' and {temp_dir_usage} deleted.")





