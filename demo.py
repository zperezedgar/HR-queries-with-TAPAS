from urllib.error import URLError
import streamlit as st
import pandas as pd
import requests
import numpy as np

API_URL = "https://api-inference.huggingface.co/models/google/tapas-large-finetuned-wtq"
headers = {"Authorization": "Bearer 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"}
csv_path = 'HRDataset_v14.csv'
question = ''
max_number_of_rows = 50  # 250

@st.cache_data
def get_data():
    df_table = pd.read_csv(csv_path)

    # convert columns to string type
    for column in df_table.columns:
        df_table[column] = df_table[column].astype("string")
        df_table[column] = df_table[column].fillna("null")

    # return df_table
    # return df_table[0:max_number_of_rows]
    return df_table.loc[0: max_number_of_rows, ['Position', 'Employee_Name', 'Salary',
                                                'PerformanceScore', 'DateofHire', 'Department', 'Absences']]


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def parse_output(df, output):
    values = []
    try:
        aggregator = output['aggregator']

        for coords in output["coordinates"]:
            value = df.iat[coords[0], coords[1]]
            if aggregator in ["SUM", "AVERAGE"]:
                value = float(value)
            values.append(value)

        if aggregator == "SUM":
            return sum(values)
        elif aggregator == "AVERAGE":
            return np.lib.function_base.average(values)
        elif aggregator == "COUNT":
            return len(values)
        else:  # "NONE"
            if len(values) == 1:
                return values[0]
            else:
                return values
    except KeyError:
        if output['error'] == 'unknown error':
            return str(output['warnings'])
        else:
            return "The model is loading..."


def post_question(q, t, dataframe):
    output = query({
        "inputs": {
            "query": q,
            "table": t
        },
     })

    # Parse answer
    return parse_output(dataframe, output)


try:
    st.write('# Company X')

    df_original = get_data()
    st.write("### Human Resources Data", df_original)

    st.write('## Data Retrieving (the traditional way)')

    df = df_original.set_index("Position")
    positions = st.multiselect(
        "Choose positions", options=set(list(df.index)), default=["Data Analyst", "Production Technician I"]
    )
    st.write('''
    df = df.set_index("Position");
    df = df.loc[positions]
    ''')

    if not positions:
        st.error("Please select at least one position.")
    else:
        data = df.loc[positions]
        st.write("### Filter by Position", data)

    # Use LM model
    table = df_original.to_dict('list')

    st.write('## Data Retrieving with NLP')

    # Get raw question from the input user
    question = st.text_input('Input your question about the table:', 'which software engineer has more absences?')

    # /////////////////////////////////////////////
    # Run the following when the user clicks a button
    # st.button("Get the answer", on_click=post_question, args=(question, table, df_original,))
    with st.form("my_form"):
        # Every form must have a submit button.
        submitted = st.form_submit_button("Get the answer")
        if submitted:
            answer = post_question(question, table, df_original)

            # Show the answer to the user
            st.write(answer)

    st.write("## Drawbacks of Table Question Answering models:")
    st.markdown("- The model may give inaccurate answers.")
    st.markdown("- This approach cannot ideally handle large tables. It becomes infeasible when the table is too "
                "large to fit in memory.")
    st.markdown("- It cannot handle databases that contain multiple tables. ")
    st.markdown("- The expressiveness is limited to a form of an aggregation over a subset of table cells."
                "Thus, structures with multiple aggregations such as "
                '_number_ _of_ _actors_ _with_ _an_ _average_ _rating_ _higher_ _than_ 4 '
                "could not be handled correctly.")
    # /////////////////////////////////////////////

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
