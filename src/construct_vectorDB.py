import os
import torch
import chromadb
import pandas as pd
from tqdm import tqdm
import utils.llm_process as llm_process
import utils.constants as constants
import utils.basic_utils as basic_utils

basic_utils.setup_logging()
logger = basic_utils.get_logger(__name__)


def vectorize_notes(center, embed_model, db_collection=False):
    try:
        logger.info("Starting vectorization process...")

        # Data
        # dataloader, dataset = basic_utils.prepare_data(center, 'all_notes', None, 32, False) # this was constructed for test set do not use it for vectorization
        center_constants = constants.CENTER_CONSTANTS[center]
        notes_df_ = basic_utils.get_data(center_constants['ALL_NOTES'], 'data')
        notes_df = notes_df_[['PAT_MRN', 'PROGRESS_NOTE']]

        # Embed model
        tokenizer, model = llm_process.initialize_local_embed_model(embed_model)
        model_max_length = model.config.max_position_embeddings

        # Setup Chroma db
        os.makedirs(constants.CHROMA_DB_PATH, exist_ok=True)
        chroma_collection_name = embed_model.split("/")[-1]
        chroma_client = chromadb.PersistentClient(path=constants.CHROMA_DB_PATH)

        if db_collection and chroma_client.get_collection(chroma_collection_name) is not None:
            logger.info(f"Deleting existing collection: '{chroma_collection_name}'...")
            chroma_client.delete_collection(chroma_collection_name)

        chroma_collection = chroma_client.get_or_create_collection(
            name=chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        for index, row in tqdm(notes_df.iterrows(), total=notes_df.shape[0], desc="Converting notes..."):
            patient_id = row['PAT_MRN']
            note = row['PROGRESS_NOTE']

            # Process the note directly without splitting
            for chunk_idx, chunk_start in enumerate(range(0, len(note), model_max_length)):
                chunk_text = note[chunk_start:chunk_start + model_max_length]
                tokens = tokenizer(chunk_text, return_tensors='pt', add_special_tokens=False)

                # Ensure the model output is handled correctly
                with torch.no_grad():  # Disable gradient calculation for inference
                    embeddings = model(tokens['input_ids'])[0][0].mean(dim=0).detach().numpy()

                # Create a unique note_id using the patient ID, row index, and chunk index
                note_id = f"{patient_id}_{index}_{chunk_idx}"  # Unique ID for each note and chunk

                # Upsert to Chroma collection
                try:
                    chroma_collection.upsert(
                        embeddings=embeddings.tolist(),
                        metadatas={
                            'pat_mrn': str(patient_id),
                            'note_id': note_id,  # Add note_id to metadata
                            'chunk_idx': chunk_idx,
                            'chunk_start': chunk_start,
                            'chunk_end': chunk_start + model_max_length,
                        },
                        documents=chunk_text,
                        ids=f"{note_id}" # Use note_id for unique ID
                    )
                    logger.info(f"Successfully upserted data for patient {patient_id}, note ID: {note_id}.")
                except Exception as e:
                    logger.error(f"Failed to upsert data for patient {patient_id}, note ID: {note_id}: {e}")


    except Exception as e:
        logger.error(f"An error occurred while processing the file: {e}")


if __name__ == "__main__":
    center = "SU"
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    vectorize_notes(center, embed_model, db_collection=False)
