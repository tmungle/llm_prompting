#!/bin/bash

# Exit if something fails
set -e

PROJ_DIR=$(dirname "$(readlink -f "$0")")
#echo "Project Directory: $PROJ_DIR"


#--DEV
#python -u "$PROJ_DIR/src/main.py" --center SU --condition 1 --model_name llama3 --prompt_type NULL_zeroShot --note_selection_mode one_note --is_include_clinical_reason True --n 1 --combine_notes False --batch_size 32
#python -u "$PROJ_DIR/src/main.py" --center SU --condition 1 --model_name gemma2 --prompt_type NULL_zeroShot --note_selection_mode one_note --is_include_clinical_reason True --n 1 --combine_notes False --batch_size 32

#--TEST
#Same sh
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama3 --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama31 --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name gemma2 --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name mistral --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name qwen --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None

python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama3 --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama31 --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name gemma2 --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name mistral --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name qwen --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None

python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama3 --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name llama31 --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name gemma2 --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name mistral --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name qwen --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process None



#python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name yi --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process 200
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name biomistral --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process 200

#1
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name med42 --prompt_type NULL_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process 200
#2
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name med42 --prompt_type Prefix_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process 200
#3
python -u "$PROJ_DIR/src/main.py" --center SU --condition 2 --model_name med42 --prompt_type InstructionBased_zeroShot --note_selection_mode all_notes --is_include_clinical_reason True --combine_notes False --batch_size 32 --max_patients_to_process 200

#python "$PROJ_DIR/src/performance_analysis.py"

echo "Bye Bye"