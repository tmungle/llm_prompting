from sphinx.addnodes import index

from utils.basic_utils import parse_args, initialize_parameters_for_condition, setup_logging, get_logger, prepare_data, get_n_value_for_note_selection_mode
import utils.llm_process as llm_process
import utils.basic_utils as basic_utils
import utils.constants as constants
from utils.constants import PROJ_DIR
import os, time, datetime

def main():

    center = "SU"
    condition = 1
    model_name = 'llama32'
    prompt_type = 'InstructionBased_zeroShot'
    n = 10
    is_include_clinical_reason = True
    note_selection_mode = 'all_notes'
    combine_notes = False
    batch_size = 32
    max_patients_to_process = None

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Experiment started")

    initialize_parameters_for_condition(condition)
    n = get_n_value_for_note_selection_mode(note_selection_mode, n)
    # logger.info(f"Running {model_name} model with {prompt_type} prompt using {n} recent note(s) for {constants.CONDITION} condition")
    logger.info(f"Running {model_name} model with {prompt_type} prompt for {note_selection_mode} using {n} recent note(s) for condition {condition} ")
    _, dataset = prepare_data(center, note_selection_mode, n, batch_size, combine_notes)
    # dataset.data.to_csv("d.csv", index=False)
    llm_process.execute_pipeline(center, dataset, condition, model_name, prompt_type, is_include_clinical_reason, note_selection_mode, n, max_patients_to_process)


    # ################# ARGS ##################
    # args = parse_args()
    # # print(args)
    # # start_time = time.time()
    # initialize_parameters_for_condition(args.condition)
    #
    # setup_logging()
    # logger = get_logger(__name__)
    # logger.info("Experiment started")
    #
    # n = get_n_value_for_note_selection_mode(args.note_selection_mode, args.n)
    # logger.info(
    #     f"Running {args.model_name} model with {args.prompt_type} prompt for {args.note_selection_mode} using {n} recent note(s) for condition {args.condition}")
    # _ , dataset = prepare_data(args.center, args.note_selection_mode, n, args.batch_size, args.combine_notes)
    # llm_process.execute_pipeline(args.center, dataset, args.condition, args.model_name, args.prompt_type, args.is_include_clinical_reason, args.note_selection_mode, n, args.max_patients_to_process)
    # end_time = time.time()

    # Save execution time
    # temp_dir = os.path.join(PROJ_DIR, 'results', 'execution_time')
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = f"{args.condition}_{args.model_name}_{args.prompt_type}_{args.is_include_clinical_reason}_{args.note_selection_mode}_{n}_{timestamp}_{args.center}"
    # output_path = os.path.join(PROJ_DIR, 'results', 'execution_time', file_name + '.txt')
    # with open(output_path, 'w') as f:
    #     f.write(f"Execution time: {round((end_time-start_time)/60, 2)} minutes\n")

    # CMD COMMAND - python S:\projects\G0002\Working\LLM_Classification\src\main.py --condition 1 --model_name llama31 --prompt_type NULL_zeroShot --note_selection_mode one_note --is_include_clinical_reason False --n 2
    # S:\projects\G0002\Working\LLM_Classification\src\main.py

    #########################################################
    # logger.info("Experiment complete\n\n")



if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    # finally:
    #     input('Press enter to exit...')
