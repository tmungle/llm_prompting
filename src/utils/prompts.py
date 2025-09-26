NULL_zeroShot = lambda is_include_clinical_reason : f"""
                    ### Task
                    Identify presence of definite {{condition}} for the given patient. 

                    ### Patient
                    Below is the patient's clinical note(s) 
                    '''
                    {{note}}
                    '''
                    ### Format your response as a JSON object with following key(s):
                    1. Status: int - 1 for definite {{condition}} OR 0 for no definite {{condition}}
                    {'2. Clinical Reason: str - your reason for definite or no definite {condition} (maximum 4 lines)' if is_include_clinical_reason else ''}

                    ### An example of how your JSON should be formatted is shown below:
                    '''json
                    {{
                        "Status" : 0/1
                        {'"Clinical Reason" : "reason1 reason 2"' if is_include_clinical_reason else ''}
                    }}
                    '''
                    The above example is only for illustration purpose only. 

                    ### Please provide your response: 
                    """


Prefix_zeroShot = lambda is_include_clinical_reason : f"""
                    ### Task
                    You are an expert ophthalmologist tasked to identify the presence of definite {{condition}} based on the clinical note(s) of a patient who visited an eye care clinic.

                    ### Patient
                    Below is the patient's clinical note(s) 
                    '''
                    {{note}}
                    '''
                    ### Format your response as a JSON object with following key(s):
                    1. Status: int - 1 for definite {{condition}} OR 0 for no definite {{condition}}
                    {'2. Clinical Reason: str - your reason for definite or no definite {condition} (maximum 4 lines)' if is_include_clinical_reason else ''}

                    ### An example of how your JSON should be formatted is shown below:
                    '''json
                    {{
                        "Status" : 0/1
                        {'"Clinical Reason" : "reason1 reason 2"' if is_include_clinical_reason else ''}
                    }}
                    '''
                    The above example is only for illustration purpose only. 

                    ### Please provide your response: 
                    """

InstructionBased_zeroShot = lambda is_include_clinical_reason : f"""
                    ### Task
                    You are an expert ophthalmologist tasked to identify the presence of definite {{condition}} based on the clinical note(s) of a patient who visited an eye care clinic.
                    
                    ### Instructions to determine whether the patient has:
                    {{condition_specific_instructions}} 
                    
                    ### Patient
                    Below is the patient's clinical note(s) 
                    '''
                    {{note}}
                    '''
                    ### Format your response as a JSON object with following key(s):
                    1. Status: int - 1 for definite {{condition}} OR 0 for no definite {{condition}}
                    {'2. Clinical Reason: str - your reason for definite or no definite {condition} (maximum 4 lines)' if is_include_clinical_reason else ''}

                    ### An example of how your JSON should be formatted is shown below:
                    '''json
                    {{
                        "Status" : 0/1
                        {'"Clinical Reason" : "reason1 reason 2"' if is_include_clinical_reason else ''}
                    }}
                    '''
                    The above example is only for illustration purpose only. 

                    ### Please provide your response: 
                    """














#     #     "Below is the clinical note(s)###NOTE START### {note} ###NOTE END###",
#     # # "glaucoma_NULL_zeroShot": " Question: Identify any presence of glaucoma from the clinical note(s) below  {note} . Provide the result in this format {Status: 0 (No Evidence) or 1 (Definite); Clinical Reason: (2 lines)}" ,
#     # # "glaucoma_NULL_zeroShot": "RETURN Status: 0 (No Evidence) or 1 (Possible) or 2 (Definite); Clinical Reason: (2 lines) any presence of glaucoma. Below is the clinical note(s)###NOTE START### {note} ###NOTE END###",
#     #
#     # "glaucoma_Prefix_zeroShot": "You are an expert ophthalmologist tasked to identify the presence of glaucoma based on the clinical note(s) of a patient who visited an eye care clinic. RETURN a JSON object with Status: 0 (No Evidence) or 1 (Definite); Explanation: (2 lines). Below is the clinical note(s)###NOTE START### {note} ###NOTE END###",
#     #
#     # "glaucoma_Instructions_based": "You are an expert ophthalmologist tasked to identify patient with glaucoma as “Definite”, “Possible” or “No Evidence” based on patient’s clinical note(s) who visited eye care clinic. Below is the clinical note(s) for the patient:###NOTE START### {note} ###NOTE END###. Instructions to determine whether the patient has (1) “Definite” glaucoma: Clinical note mentions listed problem or diagnosis of glaucoma in the assessment/plan section AND any one or more of the following: an IOP>21; a cup to disc ratio greater than or equal to 0.5; glaucoma medications; glaucoma surgery; laser treatment; type of glaucoma (Primary open-angle glaucoma (POAG), suspect, chronic angle closure, mixed mechanism, congenital, secondary, etc). Criteria should be in the same eye (that is, right eye glaucoma diagnosis and one or more of the clinical criteria above, left eye glaucoma diagnosis and one or more of the clinical criteria listed above, or both eyes glaucoma diagnosis and both eyes with one or more of the clinical criteria listed above); (2)“No Evidence” glaucoma: None of the above. RETURN a JSON object with Status: 0 (No Evidence) or 1 (Definite); Explanation: (2 lines)"
# }