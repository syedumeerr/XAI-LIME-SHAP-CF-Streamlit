import streamlit as st
import pandas as pd
import pickle

# Load the Dice Counterfactuals from the saved file
cf = pickle.load(open('D:/DS/Semester_1/MachineLearning/TermProject/dice_counterfactuals.sav', 'rb'))

def filter_counterfactuals(counterfactuals, filters):
    filtered_cf = counterfactuals

    # Apply each filter condition
    for column, value in filters.items():
        filtered_cf = filtered_cf[filtered_cf[column] == value]

    return filtered_cf

def display_counterfactuals(counterfactuals):
    if counterfactuals.empty:
        st.write("No counterfactuals found for the selected filters.")
    else:
        st.write(counterfactuals)

def main():
    st.title('Counterfactual Results')

    # Extract relevant information from the CounterfactualExamples object
    counterfactuals_data = []
    for cf_example in cf.cf_examples_list:
        cf_data = {
            'tenure': cf_example.feature1_value,
            'TotalCharges': cf_example.feature2_value,
            # Add more features as needed
            'Counterfactual': cf_example.final_cfs_df,
            'Churn': cf_example.original_outcome,
            'Counterfactual Outcome': cf_example.final_cfs_df['outcome_score'],
        }
        counterfactuals_data.append(cf_data)

    # Create a DataFrame from the extracted data
    counterfactuals_df = pd.DataFrame(counterfactuals_data)

    # Display the counterfactuals in a tabular format
    st.dataframe(counterfactuals_df)


if __name__ == '__main__':
    main()