import os
import json
import torch
import datetime
import pandas as pd
from torch.ao.nn.quantized.functional import threshold
from torch.ao.quantization import get_embedding_qat_module_mappings
from tqdm import tqdm
from transformers import pipeline
from utils import basic_utils, constants, llm_process
from typing import Dict, List, Any, Optional
from transformers import AutoModel, AutoTokenizer
import chromadb, argparse


# Initialize logging
logger = basic_utils.get_logger(__name__)


def get_embeddings(text: str, embed_model_name:str) -> torch.Tensor:
    """Embed the input text using the specified model and tokenizer."""
    tokenizer, model = llm_process.initialize_local_embed_model(embed_model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


def retrieve_relevant_docs(patient_id: int,
                      db_collection: chromadb.Collection,
                      query_instruction: str,
                      embedding_model_name: str,
                      similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Given a patient ID and instruction, query Chroma db for relevant documents.
    Return a list of documents that meet the similarity `threshold`.
    """
    # Embed the instruction
    query_embedding = get_embeddings(query_instruction, embedding_model_name).tolist()

    # Execute query on Chroma DB
    query_results: Dict = db_collection.query(
        query_embeddings=query_embedding,
        where={"pat_mrn": str(patient_id)},
        include=["metadatas", "documents", "distances"],
    )
    #
    # print(similarity_threshold)
    # print("\n", query_results)

    # Initialize a list to store filtered documents
    filtered_documents: List[Dict[str, Any]] = []

    # Iterate through the results and filter based on similarity and text length
    for doc_id, dist, doc_text, doc_metadata in zip(query_results['ids'][0], query_results['distances'][0],
                                                    query_results['documents'][0], query_results['metadatas'][0]):
        similarity_score: float = 1 - dist
        if similarity_threshold is not None and similarity_score < similarity_threshold:
            continue
        filtered_documents.append({
            'id': doc_id,
            'metadata': doc_metadata,
            'similarity': similarity_score,
            'text': doc_text,
        })
    # print("hi")
    # Sort the filtered documents by note ID and chunk index
    return sorted(filtered_documents, key=lambda x: (x['metadata']['note_id'], int(x['metadata']['chunk_idx'])))


def execute_rag_pipeline(center, dataset, condition, embedding_model_name, model_name, prompt_type,
                         is_include_clinical_reason, threshold):

    # Setup Prompting pipeline
    basic_utils.initialize_parameters_for_condition(condition)
    logger = basic_utils.get_logger(__name__)

    tokenizer, pipe, prefix_allowed_tokens_fn, prompt_instructions = llm_process.setup_pipeline(model_name,
                                                                                                is_include_clinical_reason,
                                                                                                prompt_type)
    # print(prompt_instructions)
    logger.info("Pipeline initialized")

    # Setup Retriver
    # Initialize ChromaDB client
    chromadb_path = os.path.join(constants.PROJ_DIR, "data", "chromadb")
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    logger.info("ChromaDB client initialized")

    # Load ChromaDB Collection
    chroma_collection_name = embedding_model_name.split("/")[-1]
    collection = chroma_client.get_collection(chroma_collection_name)

    batch_results = []
    final_results = []

    # Extract patient IDs from the dataset
    patient_ids = dataset.data['PAT_MRN'].unique()

    for patient_id in tqdm(patient_ids, desc="Processing patient IDs"):
        logger.info(f"***** Processing Patient ID: {patient_id} *****")

        # Get the corresponding instruction for the condition
        instruction = constants.INSTRUCTION_FAMILY.get(str(condition))
        # print(instruction)

        try:
            # logger.info(
            #     f"Retrieving notes from the database for patient ID: {patient_id} with instruction: {instruction}")

            # Generate response using RAG for the patient ID
            # Retrieve relevant notes from the vectorized database using the instruction as the query
            relevant_docs = retrieve_relevant_docs(patient_id, collection, instruction, embedding_model_name, threshold)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")

            # Combine retrieved notes into a single string
            combined_notes = " ############# \n".join([doc['text'] for doc in relevant_docs])
            # print("\n", combined_notes)

            prompt_tokens = len(tokenizer.encode(prompt_instructions))
            truncated_note, was_truncated = llm_process.truncate_notes(tokenizer, combined_notes,
                                                           max_tokens=constants.MODEL_MAX_TOKENS[
                                                                          model_name] - prompt_tokens - 64)

            task_with_note = prompt_instructions.replace("{note}", truncated_note)
            print(task_with_note)
            # print("%%%%%%%")
            try:
                generated_response, condition_status, clinical_reason = llm_process.run_llm_prompt(pipe,
                                                                                    prefix_allowed_tokens_fn,
                                                                                    task_with_note,
                                                                                    is_include_clinical_reason)
                if condition_status not in [0, 1]:
                    logger.info(f"Skipping the note for {patient_id} as condition is not 0 or 1")
                    continue

            except Exception as e:
                logger.info(f"Error in processing note for patient id {patient_id}: {e}")
                continue
            # response_tokens = len(tokenizer.encode(generated_response[0]['generated_text']))
            ########
            # rag_response = rag_pipeline(collection, embedding_model_name, model_name, prompt_type, is_include_clinical_reason, instruction,
            #                             patient_id, 0.25)
            # generated_text = rag_response[0]['generated_text'] if rag_response else None

            batch_results.append({'PAT_MRN': patient_id,
                                  # 'Task': task_with_note,
                                  'Generated_Text': json.dumps(generated_response[0]).encode().decode('unicode-escape'),
                                  'Condition_Status': condition_status,
                                  'Clinical_Reason': clinical_reason
                                  })

        except Exception as e:
            logger.info(f"Error in processing notes for patient ID {patient_id}: {e}")
            continue

    result_df = pd.DataFrame(batch_results)
    print(result_df)
    # Ensure the number of ids processed are the same as the original dataset
    # num_patients_original = set(dataset.data['PAT_MRN'].unique())
    # print(num_patients_original)
    # logger.info(f"Number of patients in original dataset: {len(num_patients_original)}")
    # num_patients_processed = (result_df['PAT_MRN'].unique())
    # print(num_patients_processed)
    # logger.info(f"Number of patients processed: {len(num_patients_processed)}")
    # missing_patients = num_patients_original - num_patients_processed

    # if missing_patients:
    #     missing_df = pd.DataFrame({
    #         'PAT_MRN': list(missing_patients),
    #         'Generated_Text': None,
    #         'Condition_Status': 0,
    #         'Clinical_Reason': None
    #     })
    #     result_df = pd.concat([result_df, missing_df], ignore_index=True)

    logger.info(f"Saving the results to results folder of the project directory")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_filename = f"RAG_{condition}_{model_name}_{prompt_type}_{is_include_clinical_reason}_{timestamp}_{center}"
    logger.info(f"File name: {result_filename}")
    output_path = os.path.join(constants.PROJ_DIR, 'results', result_filename + '.csv')
    result_df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG pipeline for retrieval.")
    parser.add_argument("--condition", type=str, default="example_condition", help="Condition to process.")
    parser.add_argument("--center", type=str, default="SU", help="Center to use.")
    parser.add_argument("--note_selection_mode", type=str, default="all_notes", help="Note selection mode.")
    parser.add_argument("--n", type=int, default=None, help="Number of recent notes to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--combine_notes", action="store_true", help="Combine notes into a single document.")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("--model_name", type=str, default="llama3", help="Model name.")
    parser.add_argument("--prompt_type", type=str, default="example_prompt", help="Prompt type.")
    parser.add_argument("--is_include_clinical_reason", action="store_true", help="Include clinical reason in prompt.")

    return parser.parse_args()

def main():
    # Parse arguments (assuming you have a function to do this)
    # args = parse_args()

    condition = 1
    center = "SU"
    note_selection_mode = "all_notes"
    n = 5  # Use only 5 recent notes
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "llama32"
    prompt_type = "NULL_zeroShot"
    is_include_clinical_reason = True
    threshold = 0.25

    basic_utils.initialize_parameters_for_condition(condition)

    # Ignore the dataloader and only keep the dataset
    _, dataset = basic_utils.prepare_data(center, note_selection_mode, n, batch_size = 8,
                                          combine_notes = False)

    execute_rag_pipeline(center, dataset, condition, embedding_model_name, model_name, prompt_type,
                         is_include_clinical_reason, threshold)



    # start_time = time.time()

    # basic_utils.initialize_parameters_for_condition(args.condition)
    # basic_utils.setup_logging()
    # logger = basic_utils.get_logger(__name__)
    # logger.info("Experiment started")
    #
    # logger.info(
    #     f"Running {args.model_name} model with {args.prompt_type} prompt for {args.note_selection_mode} using {args.n} recent note(s) for condition {args.condition}")
    #
    # # Ignore the dataloader and only keep the dataset
    # _, dataset = basic_utils.prepare_data(args.center, args.note_selection_mode, args.n, args.batch_size,
    #                                       args.combine_notes)
    #
    # execute_rag_pipeline(args.center, dataset, args.condition, args.embed_model, args.model_name, args.prompt_type,
    #                      args.is_include_clinical_reason)
    #
    # end_time = time.time()
    # logger.info(f"Experiment completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()