import torch, os
from src.utils import prompts
from src.utils import schemas


# Paths
HF_TOKEN = ""
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE_DIR = os.path.join(PROJ_DIR, 'cache')
MODEL_DIR_PATH = "/Projects/llm_prompting/models"
CHROMA_DB_PATH = os.path.join(PROJ_DIR, "data", "chromadb")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE != 'cpu':
    print(f"{DEVICE} is present")
else:
    print(f"{DEVICE} is not present")

######### LLM constants #########

MODEL_FAMILY = {
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",  # DONE
    "gemma2": "google/gemma-2-9b-it",  # DONE
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",  # DONE
    "med42": "m42-health/Llama3-Med42-8B",  # DONE
    "yi": "01-ai/Yi-1.5-9B-Chat-16K",
    "biomistral": "BioMistral/BioMistral-7B",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",  # NOT WORKING
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama32": "meta-llama/Llama-3.2-1B",

    "M_yi": "01-ai/Yi-1.5-34B-Chat-16K",  # Memory Issue for all medium size models
    "M_gemma2": "google/gemma-2-27b-it",
    "M_qwen": "Qwen/Qwen2.5-32B-Instruct",
    "M_mistral": "mistralai/Mistral-Small-Instruct-2409",

}

MODEL_MAX_TOKENS = {
    "llama31": 131072,
    "llama3": 8192,
    # "gemma": 8192,
    "gemma2": 8192,
    "mistral": 32768,
    "med42": 8192,
    "eyellama": 4096,
    "yi": 16384,
    "biomistral": 32768,
    "qwen": 32768,
    "llama32": 8192,
    # "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

PROMPT_FAMILY = {
    "NULL_zeroShot": prompts.NULL_zeroShot,
    "Prefix_zeroShot": prompts.Prefix_zeroShot,
    "InstructionBased_zeroShot": prompts.InstructionBased_zeroShot,
}

INSTRUCTION_FAMILY_ = {
    "1": f"""   Glaucoma: Clinical note mentions 
                1. Glaucoma diagnosis or glaucoma related problems AND 
                2. any one or more of the following in the affected eye(s): 
                    2.1 an intraocular pressure greater than 21; 
                    2.2 a cup to disc ratio greater than or equal to 0.5; 
                    2.3 glaucoma medications; 
                    2.4 glaucoma surgery; 
                    2.5 laser treatment; 
                    2.6 type of glaucoma. 
                Criteria should be fulfilled in the same eye (right and/or left) that is assigned the glaucoma diagnosis. That is, the patient must have a right eye glaucoma diagnosis with one or more of the clinical criteria above in the right eye, a left eye glaucoma diagnosis with one or more of the clinical criteria listed above in the left eye, or a glaucoma diagnosis in both eyes with both eyes having one or more of the clinical criteria listed)
                """,
    "2": f""" nijfnv
            """,
    "3": f""" cnwencwj
            """
}

INSTRUCTION_FAMILY = {
    "1": f"""   (A) Definite Glaucoma: Clinical note mentions 
                        1. Glaucoma diagnosis or glaucoma related problems AND  
                        2. Any one or more of the following in one or both eyes (right and/or left eye): 
                            2.1 an IOP (intraocular pressure) greater than 21; 
                            2.2 optic disc (optic nerve, sometimes simply referred to as “disc” or “disk”) with a cup to disc ratio (also referred to as cup:disc, or c:d) greater than or equal to 0.5; 
                            2.3 glaucoma medications
                                a. Latanoprost (Brand name: Latanoprost or Xalatan)
                                b. Bimatoprost (Brand name: Bimatoprost or Lumigan)
                                c. Travoprost (Brand name: Travoprost or Travatan Z)
                                d. Tafluprost (Brand name: Tafluprost or Zioptan)
                                e. Timolol (Brand name: Timolol, Timoptic, or Timoptic ocudose)
                                f. Betimol (Brand name: Betimol, Betaxolol, or Istalol)
                                g. Timolol (Brand name: Betoptic)
                                h. Carteolol (Brand name: Carteolol or Ocupress)
                                i. Levobunolol (Brand name: Levobunolol or Betagan)
                                j. Brimonidine (Brand name: Brimonidine or Alphagan)
                                k. Dorzolamide (Brand name: Dorzolamide or Trusopt)
                                l. Brinzolamide (Brand name: Brinzolamide or Azopt)
                                m. Netarsudil (Brand name: Netarsudil or Rhopressa)
                                n. Latanoprostene bunod (Brand name: Latanoprostene bunod or Vyzulta)
                                o. Dorzolamide-Timolol (Brand name: Dorzolamide-Timolol, Cosopt, Brimonidine-Timolol or Cosopt preservative free)
                                p. Brinzolamide-Brimonidine (Brand name: Simbrinza or Rhopressa Latanoprost)
                                q. Netarsudil-Latanoprost (Brand name: Rocklatan)
                                r. Pilocarpine (Brand name: Isopto Carpine)
                                s. Acetazolamide (Brand name: Diamox)
                                t. Methazolamide (Brand name: Neptazane)
                            2.4 glaucoma surgery
                                a. Trabeculectomy (or trab)
                                b. Filtering surgery (or glaucoma filtering surgery) 
                                c. Tube (or tube shunt)
                                d. Ahmed valve
                                e. Baerveldt implant
                                f. Express shunt
                                g. Xen gel stent
                                h. iStent
                                i. Hydrus
                                j. Omni
                                k. KDB
                                l. GATT
                                m. Trabectome
                                n. Trabeculotomy
                                o. Canaloplasty
                                p. Goniotomy
                                q. MIGS (minimally invasive glaucoma surgery, or microinvasive glaucoma surgery, or micro incisional glaucoma surgery)
                                r. Cyclophotocoagulation (CPC) 
                            2.5 laser treatment
                                a. SLT (selective laser trabeculoplasty)
                                b. LPI (laser peripheral iridotomy)
                                c. ALT (argon laser trabeculoplasty)
                                d. trabeculoplasty
                            2.6 type of glaucoma.
                                a. POAG or OAG (Primary open angle glaucoma)
                                b. JOAG (juvenile open angle glaucoma)
                                c. ACG (angle closure glaucoma)
                                d. AACG (acute angle closure glaucoma)
                                e. CACG (chronic angle closure glaucoma)
                                f. Pigmentary glaucoma 
                                g. Traumatic glaucoma
                                h. Pseudoexfoliation glaucoma (PEX, PXF, or PXE) or Exfoliation glaucoma
                                i. Normal tension glaucoma (or low tension glaucoma, LTG, NTG)
                                j. Secondary glaucoma
                                k. Uveitic glaucoma
                                l. Congenital glaucoma
                                m. Neovascular glaucoma (NVG)
                                n. Phacomorphic glaucoma
                                o. Phacolytic glaucoma
                                p. Phacoanaphylactic glaucoma
                        (B) No Definite Glaucoma: Clinical note mentions
                                1.	None of the criteria listed in section (A) are met OR
                                2.	Only mentions “possible” or “suspected” for glaucoma or be screened for glaucoma without confirm diagnosis glaucoma, such patient should be classified as having “No Definite Glaucoma”

                """,
    "2": f"""   (A)	Definite Diabetic Retinopathy: Clinical note mentions any of the following in one or both eyes (right and/or left eye):
                        1.	Problem list, impression, assessment, and/or diagnosis of 
                            1.1. diabetic retinopathy (also written as DR) or 
                            1.2. non-proliferative diabetic retinopathy (also written as NPDR) or
                            1.3. proliferative diabetic retinopathy (also written as PDR) or 
                            1.4. diabetic macular edema (also written as DME). DME may also be described as CSDME (or clinically-significant diabetic macular edema)
                        OR
                        2. a history of diabetes or diabetic retinopathy with one or more of the following findings
                            2.1. microaneurysms (MAs)
                            2.2. intraretinal hemorrhages (IRHs) or dot blot hemorrhages (DBHs) or vitreous hemorrhage (VH) or flame hemorrhages 
                            2.3. exudates or cotton wool spots
                            2.4. cystoid macular edema
                            2.5. neovascularization of the iris (NVI) or neovascularization of the disc (NVD) or neovascularization elsewhere (NVE) or neovascularization of the angle (NVA)

                (B)	No Definite Diabetic Retinopathy: Clinical note mentions
                        1. None of the criteria listed in section (A) are met OR
                        2. Only mentions “possible” or “suspected” for diabetic retinopathy or be screened for diabetic retinopathy without confirm diagnosis of diabetic retinopathy (or diabetic retinopathy exam findings with a history of diabetes), such patient should be classified as having “No Definite Diabetic Retinopathy”
            """,
    "3": f"""   (A) Definite age-related macular degeneration: Clinical note mentions any or the following in one or both eyes (right and/or left eye):
                        1. problem list, impression, assessment, and/or diagnosis of age-related macular degeneration (also written as AMD or ARMD) or indication of drusen or druse (or drusenoid material), intraretinal fluid (IRF), subretinal fluid (SRF), choroidal neovascular membrane (CNVM, CNV, or choroidal neovascularization), and/or subretinal hyperreflective material (SRHM) 
                        OR
                        2. dry (also called non-exudative) or wet (also called exudative) age-related macular degeneration (note: it is possible for a patient to have dry AMD in one eye and wet AMD in the other eye. 

                (B) No Definite age-related macular degeneration: Clinical note mentions
                        1. None of the criteria listed in section A are met OR
                        2. Only mentions “possible” or “suspected” age-related macular degeneration, or be screened for age-related macular degeneration without confirm diagnosis of age-related macular degeneration, such patient should be classified as having “no definite age-related macular degeneration”
            """
}

SCHEMA_FAMILY = {
    "status_only": schemas.status_only_schema,
    "status_n_reason": schemas.status_n_reason_schema,
}

######### File constants #########

CENTER_CONSTANTS = {
    "SU": {
        "ALL_NOTES": "SU_NOTES_till_RecentVisit.csv",
        "INPUT_FILE": "test_GLA_DEPT_GRADE_SU.csv",
    },
    "UoM": {
        "ALL_NOTES": "UoM_NOTES_till_RecentVisit.csv",
        "INPUT_FILE": "test_DR_DEPT_GRADE_UoM.csv",
        # "INPUT_FILE": "test_GLA_UoM.csv",
    },
}
# CENTER = "SU"
# ALL_NOTES is the overall data
# ALL_NOTES = "UoM_NOTES_till_RecentVisit.csv"
# ALL_NOTES = "SU_NOTES_till_RecentVisit.csv"

# INPUT_FILE is pat_ids that you want to analyze
# INPUT_FILE = "test_GLA_DEPT_GRADE_SU.csv"
# INPUT_FILE = "dev_GLA_DEPT_GRADE_UoM.csv"

CONDITION = None
CONDITION_SPECIFIC_INSTRUCTION = None
true_label_variable = -999
