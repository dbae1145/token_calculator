import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import tiktoken

def main():
    st.title("Token Calculator")

    if 'text_history' not in st.session_state:
        st.session_state.text_history = []
    if 'cost_history' not in st.session_state:
        st.session_state.cost_history = []

    # Variables to store token counts
    text_tokens, prompt_tokens, user_example_tokens, assistant_example_tokens = 0, 0, 0, 0

    # Sidebar entries
    st.sidebar.header("English Text to Translate")
    text = st.sidebar.text_area("Enter your English text for translation")

    st.sidebar.header("User System Prompt")
    user_prompt = st.sidebar.text_area("Enter your prompt here")

    st.sidebar.header("Example (User)")
    user_example = st.sidebar.text_area("Enter your example for User here")

    st.sidebar.header("Example (Assistant)")
    assistant_example = st.sidebar.text_area("Enter your example for Assistant here")

    # Submit button on the sidebar
    submit = st.sidebar.button("Submit")

    # When submit is clicked, update all the calculations
    if submit:
        if text:
            st.session_state.text_history.append(text)
            text_tokens = tiktoken_len(text)

        if user_prompt:
            user_prompt_non_eng_prop = count_non_english_letters(user_prompt)/len(user_prompt)
            prompt_tokens = tiktoken_len(user_prompt)*user_prompt_non_eng_prop*2.18 + tiktoken_len(user_prompt)*(1-user_prompt_non_eng_prop)

        if user_example:
            user_example_tokens = tiktoken_len(user_example)

        if assistant_example:
            assistant_example_non_eng_prop = count_non_english_letters(assistant_example) / len(assistant_example)
            assistant_example_tokens = tiktoken_len(assistant_example) * assistant_example_non_eng_prop * 2.18 + tiktoken_len(assistant_example) * (1 - assistant_example_non_eng_prop)

        if len(st.session_state.text_history) > 1:
            prev_text_input_token = tiktoken_len(st.session_state.text_history[-2])
            prev_text_completion_token = prev_text_input_token*3.31
        else:
            prev_text_input_token = 0
            prev_text_completion_token = 0

        total_prompt_token = text_tokens + (prev_text_input_token + prev_text_completion_token) + prompt_tokens + user_example_tokens + assistant_example_tokens
        # Update the total cost history
        total_cost = total_prompt_token/1000*0.06 + text_tokens*3.31/1000*0.12
        st.session_state.cost_history.append(total_cost)

        # Create a dataframe for the results table
        data = {
            "Type": ["Current Text Input", "GPT Completion for Current Text Input (Expected)", "Previous Text Input","GPT Completion for Previous Text Input (Expected)", "User Prompt", "Example (User)", "Example (Assistant)"],
            "Estimated Token Size": [round(text_tokens), round(text_tokens*3.31), round(prev_text_input_token), round(prev_text_completion_token), round(prompt_tokens), round(user_example_tokens), round(assistant_example_tokens)]
        }
        results_df = pd.DataFrame(data)

        # Display the table using Streamlit
        st.header("Token Breakdown")
        st.table(results_df)

        st.write(f"Based on the prompt token size {round(total_prompt_token)} and the completion token size {round(text_tokens*3.31)}, the estimated cost to run the gpt-4-32k model for the translation is USD {'{:.2f}'.format(total_cost)} ({round(total_prompt_token)}/1000 x USD 0.06 + {round(text_tokens*3.31)}/1000 x USD 0.12)")

    # Display Cost History as a dataframe
    st.header("Cost History")
    cost_df = pd.DataFrame(st.session_state.cost_history, columns=["Past Costs (USD)"])

    # Append total row to the dataframe
    total_cost_row = pd.DataFrame([cost_df["Past Costs (USD)"].sum()], columns=["Past Costs (USD)"], index=["Total"])
    cost_df = pd.concat([cost_df, total_cost_row])

    st.table(cost_df)

    # Display the Text History
    st.header("Text History")
    for past_text in st.session_state.text_history:
        st.write(past_text)
        st.write("__________________________________________________________________________")

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def token_calculator_from_bytes(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)
    text = ''.join(page.get_text("text") for page in doc)
    tokens = tiktoken_len(text)
    return tokens

def count_non_english_letters(s):
    english_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    count = sum(1 for char in s if char not in english_letters)
    return count

if __name__ == "__main__":
    main()



