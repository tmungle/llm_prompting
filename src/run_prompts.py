import pandas as pd

from src.utils import llm_process
from src.utils.constants import MODEL_FAMILY, INPUT_FILE, CONDITION
from src.utils import basic_utils
from src.utils.prompts import prompt_templates


def Prompting(model_name, prompt_type):
    """
    Executes the desired task and saves the output in a CSV file.

    Parameters:
    model_name (str): A key from the MODEL_FAMILY dictionary representing a specific model.
    prompt_type (str): The key used to access the user_content from the pre-defined dictionary.
    """

    tokenizer, model = llm_process.initialize_local_model(MODEL_FAMILY[model_name])

    df = basic_utils.get_data(INPUT_FILE)

    result_df = pd.DataFrame(columns=['PAT_MRN', 'Generated_Text', 'Condition_Status'])

    for index, row in df.iloc[:5].iterrows():  # df.iloc[:10].iterrows(): #df.iterrows():

        note = row['PROGRESS_NOTE']
        user_content_ = prompt_templates[prompt_type]
        # system_message = f"{user_content} \n {note}"
        user_content = user_content_.replace("{condition}", CONDITION)
        system_message = user_content.replace("{note}", note)
        # print(system_message)

        if model_name in ["llama3", "med42"]:
            prompt = [llm_process.system(system_message)]
            outputs = llm_process.get_llama3_response(model, tokenizer, prompt)
            generated_response = outputs[0]['generated_text']
            # print(generated_response)
            condition_status = llm_process.process_response(model_name, generated_response)
            # print(condition_status)
            result_df.loc[index] = [row['PAT_MRN'], generated_response, condition_status]
        if model_name == "mistral":
            prompt = [llm_process.user(system_message)]
            generated_response = llm_process.get_mistral_response(model, tokenizer, prompt)
            # print(generated_response)
            condition_status = llm_process.process_response(model_name, generated_response)
            # print(condition_status)
            result_df.loc[index] = [row['PAT_MRN'], generated_response, condition_status]
        if model_name == "gemma":
            prompt = [llm_process.user(system_message)]
            outputs = llm_process.get_gemma_response(model, tokenizer, prompt)
            condition_status = llm_process.process_response(model_name, outputs)
            # print(condition_status)
            result_df.loc[index] = [row['PAT_MRN'], outputs, condition_status]

        print("******************************")

    print(f"Saving the results to results folder of the project directory")
    result_df.to_csv(f'../results/{CONDITION}_{model_name}_{prompt_type}.csv', index=False)


if __name__ == "__main__":
    model_name = 'llama3'
    prompt_type = 'NULL_zeroShot'
    Prompting(model_name, prompt_type)
