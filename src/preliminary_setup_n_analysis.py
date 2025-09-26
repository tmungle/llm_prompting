import os

import matplotlib.pyplot as plt
import pandas as pd
from src.utils import basic_utils, constants
# from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("classic")
sns.set_style('whitegrid', {'axes.grid': False})
palette = {}
def plot_boxplot(df, group_col, value_col, title, x_label, y_label):

    plt.figure(figsize=(8,6))
    ax = sns.boxplot(x=group_col, y=value_col, data=df, color='white', showfliers=False, width=0.15)
    plt.xticks([0,1], ["DEFINITE", "NOT DEFINITE"])
    plt.title(title)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"../results/figures/{title}", dpi=300)
    # plt.show()

def create_dev_n_test_set(center, df, outcome_var, n):
    df_positive = df[df[outcome_var] == "DEFINITE"]
    df_negative = df[df[outcome_var] == "NO EVIDENCE"]

    if len(df_positive) < n or len(df_negative) < n:
        raise ValueError(f'Not enough samples for specified outcome variable {outcome_var}')

    df_positive_prompt_dev = df_positive.sample(n=n, random_state=42)
    df_negative_prompt_dev = df_negative.sample(n=n, random_state=42)

    prompt_dev_set = pd.concat([df_positive_prompt_dev, df_negative_prompt_dev])

    prompt_dev_set_ids = prompt_dev_set.index
    test_df = df.drop(index=prompt_dev_set_ids)

    prompt_dev_set.to_csv(os.path.join(constants.PROJ_DIR, 'data', f"dev_{outcome_var}_{center}.csv"), index=False)
    test_df.to_csv(os.path.join(constants.PROJ_DIR, 'data', f"test_{outcome_var}_{center}.csv"), index=False)

    print(f"Development and test set created")

def get_stats(center, CONDITION, true_label_variable):

    if CONDITION == "glaucoma":
        label_type = "GLA"
        label_name = "Glaucoma"
    elif CONDITION == "diabetic retinopathy":
        label_type = "DR"
        label_name = "Diabetic Retinopathy"
    else:
        label_type = "AMD"
        label_name = "Age-related Macular Degeneration"

    df_labels = basic_utils.get_data(f"{center}_{label_type}_labels.csv", 'data')
    center_constants = basic_utils.get_center_constants(center)
    notes_df = basic_utils.get_data(center_constants['ALL_NOTES'], 'data')

    merged_df_ = pd.merge(df_labels, notes_df, on='PAT_MRN', how="inner")
    merged_df = merged_df_[["PAT_MRN", "PROGRESS_NOTE", true_label_variable]]
    # print(df_labels.shape, df_labels.columns)
    # print(notes_df.shape, notes_df.columns)
    # print(merged_df.shape, merged_df.columns)
    # print(df_labels['PAT_MRN'].nunique())
    # print(notes_df['PAT_MRN'].nunique())
    # print(merged_df['PAT_MRN'].nunique())
    # print(merged_df['PROGRESS_NOTE'].isnull().sum())
    # print(Counter(merged_df["GLA_DEPT_GRADE"]))

    ######## PATIENT ###########
    # Get total number of patients
    print("\nTotal number of patients: ", merged_df["PAT_MRN"].nunique())

    # Get median and IQRs notes for total number of patients
    total_notes = merged_df.groupby("PAT_MRN")["PROGRESS_NOTE"].count()
    print("\nMedian IQR Notes: ", total_notes.median(), "[", total_notes.quantile(0.25), "-", total_notes.quantile(0.75),
          "]")

    # Get word count
    total_words = merged_df["PROGRESS_NOTE"].str.split().str.len()
    print("\nMedian IQR Words: ", total_words.median(), "[", total_words.quantile(0.25), "-",
          total_words.quantile(0.75), "]")

    ####### LABELS ########
    print("####### LABELS ########")
    # Get number of patients for each label
    print("\nTotal number of patients across each label", merged_df.groupby(true_label_variable)["PAT_MRN"].nunique())

    overall_note_counts_ = merged_df.groupby([true_label_variable, "PAT_MRN"])["PROGRESS_NOTE"].count()
    # print(overall_note_counts)
    overall_note_counts = overall_note_counts_.groupby(true_label_variable)
    print(f"\nMedian IQR Notes: ", overall_note_counts.median(), "[", overall_note_counts.quantile(0.25), "-",
          overall_note_counts.quantile(0.75), "]")
    plot_boxplot(df=overall_note_counts_.reset_index(), group_col=true_label_variable, value_col="PROGRESS_NOTE",
                 title=f"Boxplot of {center} {label_name} Note Counts", x_label= label_name, y_label="Number of Notes")

    # Get word count for each label

    # median_words = merged_df.groupby(true_label_variable)["PROGRESS_NOTE"].apply(lambda x: x.str.split().str.len().median())
    # iqr_25_words = merged_df.groupby(true_label_variable)["PROGRESS_NOTE"].apply(lambda x: x.str.split().str.len().quantile(0.25))
    # iqr_75_words = merged_df.groupby(true_label_variable)["PROGRESS_NOTE"].apply(lambda x: x.str.split().str.len().quantile(0.75))
    # print(median_words, iqr_25_words, iqr_75_words)

    # alternative
    grouped_word_counts_ = merged_df.groupby(true_label_variable)["PROGRESS_NOTE"].apply(lambda x: x.str.split().apply(len))
    grouped_word_counts = grouped_word_counts_.groupby(true_label_variable)
    print("\nTotal number of words for each label: ", grouped_word_counts.median(), "25% IQR: ",
          grouped_word_counts.quantile(0.25), "75% IQR: ", grouped_word_counts.quantile(0.75))

    # box plot
    temp_df = merged_df
    temp_df["word_count"] = temp_df["PROGRESS_NOTE"].str.split().str.len()
    words_per_group = temp_df.groupby(true_label_variable)["word_count"].apply(list).reset_index()
    words_per_group_df = words_per_group.explode("word_count")
    plot_boxplot(df=words_per_group_df, group_col=true_label_variable, value_col="word_count",
                 title=f"Boxplot of {center} {label_name} Word Count", x_label=label_name, y_label="Number of Words")


def main():
    center = "SU"
    value = 1  # 1 - glaucoma
    basic_utils.initialize_parameters_for_condition(value)

    ####### Create train and test set #######
    # df = basic_utils.get_data("UoM_AMD_labels.csv", 'data')
    # n = 20
    # PLEASE DO NOT RUN THIS EVERYTIME while running the script
    # create_dev_n_test_set(center, df, constants.true_label_variable, n)

    ####### Get stats #########
    get_stats(center, constants.CONDITION, constants.true_label_variable)

if __name__ == "__main__":
    main()
