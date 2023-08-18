import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import tiktoken
import PyPDF2
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import plotly.graph_objects as go
from bs4 import BeautifulSoup

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def gpt_translator(gpt_model, text_to_translate, user_prompt, temperature, token_limit):
    response = openai.ChatCompletion.create(
        engine=gpt_model,
        messages=[
            {'role': 'system', 'content': user_prompt},
            # {'role':'user', 'content': example_user},
            # {'role': 'user', 'content': example_assistant},
            {'role': 'user', 'content': text_to_translate}
        ],
        temperature=temperature,
        max_tokens=round(token_limit - tiktoken_len(text_to_translate)-tiktoken_len(user_prompt)),
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response['choices'][0]['message']['content'], response['usage']['prompt_tokens'], response['usage']['completion_tokens']

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


def merge_rows_in_column(html, column_name):
    soup = BeautifulSoup(html, 'html.parser')

    # Find the index of the target column
    headers = [th.get_text() for th in soup.findAll('th')]
    column_index = headers.index(column_name)

    # Get all the rows in the table
    rows = soup.findAll('tr')[1:]  # Exclude header row

    # Initialize the previous value to the first cell's value
    prev_value = rows[0].findAll('td')[column_index].get_text()

    # Initialize rowspan
    rowspan = 1

    # Start from the second row
    for i in range(1, len(rows)):
        current_value = rows[i].findAll('td')[column_index].get_text()

        # If the value is the same as the previous one, increase the rowspan
        if current_value == prev_value:
            rowspan += 1
            # Remove the current cell so it appears merged
            rows[i].findAll('td')[column_index].decompose()
        else:
            # Set the rowspan attribute to the previous cell
            rows[i - rowspan].findAll('td')[column_index]['rowspan'] = rowspan
            rowspan = 1  # Reset the rowspan
            prev_value = current_value

    # Handle the case where the last group of rows has the same value
    if rowspan > 1:
        rows[-rowspan].findAll('td')[column_index]['rowspan'] = rowspan

    return str(soup)

def cost_calculator():
    st.title("Cost Calculator")

    option = st.sidebar.selectbox('How would you like to bring your text for translation?', ('Import PDF', 'Text'))

    if option == 'Import PDF':
        st.sidebar.header("Upload your PDF")
        uploaded_pdfs = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True) #200MB per file
    else:
        st.sidebar.header("Text to Translate")
        text_to_translate = st.sidebar.text_area("Enter your text here", value=st.session_state.get('text_to_translate', ''))
        st.session_state.text_to_translate = text_to_translate

    st.sidebar.header("System Prompt")
    user_prompt = st.sidebar.text_area("Enter your prompt here", value=st.session_state.get('user_prompt', ''))
    st.session_state.user_prompt = user_prompt

    st.sidebar.header("Example (User)")
    user_example = st.sidebar.text_area("Enter your example for User here",value=st.session_state.get('user_example', ''))
    st.session_state.user_example = user_example

    st.sidebar.header("Example (Assistant)")
    assistant_example = st.sidebar.text_area("Enter your example for Assistant here", value=st.session_state.get('assistant_example', ''))
    st.session_state.assistant_example = assistant_example

    if 'refine_translation' not in st.session_state:
        st.session_state.refined_translation = False
    refined_translation = st.sidebar.checkbox("Perform Refined Translation within the Same Language", value=st.session_state.refined_translation)
    st.session_state.refined_translation = refined_translation

    # Submit button on the sidebar
    submit = st.sidebar.button("Submit")

    total_cost_dict = {
            'File': [],
            'Page Count': [],
            'Number of Translation Blocks': [],
            'Recommended GPT Model': [],
            'Estimated Translation Cost (USD)': []
        }

    chunks_dict = {}

    # Compute new data if submit is pressed
    if submit:
        if option == 'Import PDF':
            for uploaded_pdf in uploaded_pdfs:
                pdf_name = uploaded_pdf.name
                pdf_data = uploaded_pdf.getvalue()

                with BytesIO(pdf_data) as open_pdf:
                    pdf = PyPDF2.PdfReader(open_pdf)
                    text = "".join(page.extract_text() + ' ' for page in pdf.pages)
                    page_num = len(pdf.pages)

                prompt_tokens = tiktoken_len(st.session_state.user_prompt)
                user_example_tokens = tiktoken_len(st.session_state.user_example)
                assistant_example_tokens = tiktoken_len(st.session_state.assistant_example)

                total_prompt_token = prompt_tokens + user_example_tokens + assistant_example_tokens

                gpt_token_limit = 8000

                token_room = gpt_token_limit * 0.9 - total_prompt_token

                if token_room > 0:
                    gpt_model = 'gpt-4-8k'
                    prompt_cost = 0.03
                    completion_cost = 0.06
                else:
                    gpt_model = 'gpt-4-32k'
                    gpt_token_limit = 32000
                    token_room = gpt_token_limit * 0.9 - total_prompt_token
                    prompt_cost = 0.06
                    completion_cost = 0.12

                st.session_state.recommended_model = gpt_model

                chunk_token_limit = token_room / 4 #why am I dividing this by 4 (out of tokens left, 1 to the chunk size and 3 to the translated outcome)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_token_limit,
                    chunk_overlap=0,  # number of tokens overlap between chunks
                    length_function=tiktoken_len,
                    separators=['\n\n', '\n', ' ', '']
                )

                chunks = text_splitter.split_text(text)

                total_estimated_cost = 0
                for i, chunk in enumerate(chunks):
                    chunk_size = tiktoken_len(chunk)
                    estimated_prompt_tokens = total_prompt_token + chunk_size

                    if refined_translation:
                        estimated_completion_tokens = chunk_size
                    else:
                        estimated_completion_tokens = chunk_size*2

                    estimated_cost = estimated_prompt_tokens / 1000 * prompt_cost + estimated_completion_tokens / 1000 * completion_cost
                    total_estimated_cost += estimated_cost

                total_cost_dict['File'].append(pdf_name)
                total_cost_dict['Page Count'].append(page_num)
                total_cost_dict['Number of Translation Blocks'].append(len(chunks))
                total_cost_dict['Recommended GPT Model'].append(gpt_model)
                total_cost_dict['Estimated Translation Cost (USD)'].append(total_estimated_cost)

                chunks_dict[pdf_name] = chunks #chunks in a list

            total_cost_df = pd.DataFrame(total_cost_dict)
            total_row = {
                'File':['Total'],
                'Page Count': [total_cost_df['Page Count'].sum()],
                'Number of Translation Blocks': [total_cost_df['Number of Translation Blocks'].sum()],
                'Recommended GPT Model': [list(set(total_cost_df['Recommended GPT Model']))[0]],
                'Estimated Translation Cost (USD)': [total_cost_df['Estimated Translation Cost (USD)'].sum()]

            }
            total_df = pd.DataFrame(total_row)
            total_cost_df =  pd.concat([total_cost_df, total_df])
            total_cost_df['Estimated Translation Cost (USD)'] = total_cost_df['Estimated Translation Cost (USD)'].apply(lambda x: '${:.2f}'.format(x))

            html = total_cost_df.to_html(index=False)

            merged_html = merge_rows_in_column(html, 'Recommended GPT Model')

            st.session_state.calculated_data = {
                'total_cost_estimate': merged_html,
                'chunks': chunks_dict
            }
        else:
            prompt_tokens = tiktoken_len(st.session_state.user_prompt)
            user_example_tokens = tiktoken_len(st.session_state.user_example)
            assistant_example_tokens = tiktoken_len(st.session_state.assistant_example)

            total_prompt_token = prompt_tokens + user_example_tokens + assistant_example_tokens

            gpt_token_limit = 8000

            token_room = gpt_token_limit * 0.9 - total_prompt_token

            if token_room > 0:
                gpt_model = 'gpt-4-8k'
                prompt_cost = 0.03
                completion_cost = 0.06
            else:
                gpt_model = 'gpt-4-32k'
                gpt_token_limit = 32000
                token_room = gpt_token_limit * 0.9 - total_prompt_token
                prompt_cost = 0.06
                completion_cost = 0.12

            st.session_state.recommended_model = gpt_model

            chunk_token_limit = token_room / 4  # why am I dividing this by 4 (out of tokens left, 1 to the chunk size and 3 to the translated outcome)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_token_limit,
                chunk_overlap=0,  # number of tokens overlap between chunks
                length_function=tiktoken_len,
                separators=['\n\n', '\n', ' ', '']
            )

            chunks = text_splitter.split_text(st.session_state.text_to_translate)

            total_estimated_cost = 0
            for i, chunk in enumerate(chunks):
                chunk_size = tiktoken_len(chunk)
                estimated_prompt_tokens = total_prompt_token + chunk_size

                if refined_translation:
                    estimated_completion_tokens = chunk_size
                else:
                    estimated_completion_tokens = chunk_size * 2

                estimated_cost = estimated_prompt_tokens / 1000 * prompt_cost + estimated_completion_tokens / 1000 * completion_cost
                total_estimated_cost += estimated_cost

            total_cost_dict['File'].append(st.session_state.text_to_translate)
            total_cost_dict['Page Count'].append('N/A')
            total_cost_dict['Number of Translation Blocks'].append(len(chunks))
            total_cost_dict['Recommended GPT Model'].append(gpt_model)
            total_cost_dict['Estimated Translation Cost (USD)'].append(total_estimated_cost)

            chunks_dict['Text'] = chunks  # chunks in a list

        total_cost_df = pd.DataFrame(total_cost_dict)

        total_cost_df['Estimated Translation Cost (USD)'] = total_cost_df['Estimated Translation Cost (USD)'].apply(lambda x: '${:.2f}'.format(x))

        html = total_cost_df.to_html(index=False)

        st.session_state.calculated_data = {
            'total_cost_estimate': html,
            'chunks': chunks_dict
        }
    # Divide the page using columns with a 2:1 ratio
    col1, col2 = st.columns((1, 1))

    # Use the stored calculations if available
    if 'calculated_data' in st.session_state:
        with col1:
            col1.markdown("### Summary")

            # col1.dataframe(st.session_state.calculated_data['total_cost_estimate'], hide_index=True)
            col1.write(st.session_state.calculated_data['total_cost_estimate'], unsafe_allow_html=True)

            # col1.markdown("### GPT Model")
            # col1.markdown("""
            # gpt-4-8k --> "Short Story Translator"
            # gpt-4-32k --> "Long Story Translator"
            #
            # The **gpt-4-8k** is like a small book that can translate up to 600 words. Great for brief articles or quick chats! The **gpt-4-32k** is like a big novel, designed for translating up to 2,400 words. It's ideal for in-depth reports or long stories.
            #
            # If you're translating a short piece, the 'Short Story Translator' might be just what you need. But for bigger projects with lots of details, you'll want to use the 'Long Story Translator'. Translating longer pieces is pricier.""")

        with col2:
            col2.markdown("### Translation Blocks")
            option = col2.selectbox(
                'Select a file',
                tuple(st.session_state.calculated_data['chunks'].keys())

            )

            if option:
                for i, chunk in enumerate(st.session_state.calculated_data['chunks'][option]):
                    col2.markdown(f"**Block {i + 1}**")
                    col2.write(f"<i>{chunk}</i>", unsafe_allow_html=True)
                    col2.write("________________________________________________________________________")

    # Add the "Clear" button below the "Submit" button
    clear = st.sidebar.button("Clear All")

    # If "Clear" button is pressed, reset the session state and rerun the app
    if clear:
        keys_to_clear = [
            'gpt_model', 'pdf_data', 'pdf_text', 'user_prompt',
            'user_example', 'assistant_example', 'calculated_data'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.experimental_rerun()

def translator():
    st.title("Translator")

    st.sidebar.header("Azure OpenAI Credentials")
    azure_openai_api_base = st.sidebar.text_input("Enter your Azure OpenAI API base here", value=st.session_state.get('azure_openai_api_base', ''), type='password')
    st.session_state.azure_openai_api_base = azure_openai_api_base

    azure_openai_api_version = st.sidebar.text_input("Enter your Azure OpenAI API version here", value=st.session_state.get('azure_openai_api_version', ''), type='password')
    st.session_state.azure_openai_api_version = azure_openai_api_version

    azure_openai_api_key = st.sidebar.text_input("Enter your Azure OpenAI API key here", value=st.session_state.get('azure_openai_api_key', ''),type='password')
    st.session_state.azure_openai_api_key = azure_openai_api_key

    openai.api_type = "azure"
    openai.api_base = st.session_state.azure_openai_api_base
    openai.api_version = st.session_state.azure_openai_api_version
    openai.api_key = st.session_state.azure_openai_api_key

    st.sidebar.header("Upload your Document")
    uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)  # 200MB per file

    language = st.sidebar.selectbox('Choose destination language', ('Korean', 'Chinese', 'Japanese', 'Russian', 'French', 'Arabic'))
    st.session_state.language = language

    st.sidebar.header("System Prompt")
    user_prompt = st.sidebar.text_area("Enter your prompt here", value=st.session_state.get('user_prompt', ''))
    st.session_state.user_prompt = user_prompt

    # st.sidebar.header("Example (User)")
    # user_example = st.sidebar.text_area("Enter your example for User here",value=st.session_state.get('user_example', ''))
    # st.session_state.user_example = user_example
    #
    # st.sidebar.header("Example (Assistant)")
    # assistant_example = st.sidebar.text_area("Enter your example for Assistant here", value=st.session_state.get('assistant_example', ''))
    # st.session_state.assistant_example = assistant_example

    cost_calculate = st.sidebar.button("Calculate Cost")

    if cost_calculate:
        pdf_name = uploaded_pdf.name
        pdf_data = uploaded_pdf.getvalue()

        with BytesIO(pdf_data) as open_pdf:
            pdf = PyPDF2.PdfReader(open_pdf)
            text = "".join(page.extract_text() + ' ' for page in pdf.pages)
            page_num = len(pdf.pages)

        prompt_tokens = tiktoken_len(st.session_state.user_prompt)
        # example_user_tokens = tiktoken_len(st.session_state.user_example)
        # example_assistant_tokens = tiktoken_len(st.session_state.assistant_example)

        total_prompt_token = prompt_tokens

        gpt_token_limit = 8000

        token_room = gpt_token_limit * 0.9 - total_prompt_token

        if token_room > 0:
            gpt_model = 'gpt-4-8k'
            prompt_cost = 0.03
            completion_cost = 0.06
        else:
            gpt_model = 'gpt-4-32k'
            gpt_token_limit = 32000
            token_room = gpt_token_limit * 0.9 - total_prompt_token
            prompt_cost = 0.06
            completion_cost = 0.12

        st.session_state.recommended_model = gpt_model

        chunk_token_limit = token_room / 4  # why am I dividing this by 4 (out of tokens left, 1 to the chunk size and 3 to the translated outcome)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_token_limit,
            chunk_overlap=0,  # number of tokens overlap between chunks
            length_function=tiktoken_len,
            separators=['\n\n', '\n', ' ', '']
        )

        chunks = text_splitter.split_text(text)
        st.session_state.chunks = chunks
        st.write(f"Total Number of Chunks: {len(st.session_state.chunks)}")
        total_estimated_cost = 0
        for i, chunk in enumerate(chunks):
            chunk_size = tiktoken_len(chunk)
            estimated_prompt_tokens = total_prompt_token + chunk_size

            estimated_completion_tokens = chunk_size * 2

            estimated_cost = estimated_prompt_tokens / 1000 * prompt_cost + estimated_completion_tokens / 1000 * completion_cost
            total_estimated_cost += estimated_cost

        st.session_state.total_cost_estimate = total_estimated_cost

    if 'total_cost_estimate' not in st.session_state:
        st.session_state.total_cost_estimate = 0
    st.sidebar.write(f"Estimated Cost (USD): {'${:.2f}'.format(st.session_state.total_cost_estimate)}")

    col1, col2 = st.columns((1, 1))

    with col1:
        col1.markdown("### Translation Style")
        style = st.select_slider(
            'Choose a translation style',
            options=['Creative', 'Balanced', 'Precise'],
            value=st.session_state.get('selected_translation_style', 'Balanced')  # Default value if not in session_state
        )
        st.session_state.selected_translation_style = style

        if style == 'Creative':
            st.session_state.temperature = 0.7
        elif style == 'Balanced':
            st.session_state.temperature = 0.4
        else:
            st.session_state.temperature = 0.1

        col1.markdown("<br/>", unsafe_allow_html=True)
        col1.markdown("### GPT Model")

        models = ["gpt-4-8k", "gpt-4-32k"]
        if 'recommended_model' in st.session_state:
            col1.markdown(f"**Recommended Model:** {st.session_state.recommended_model}")
        selected_model = st.selectbox(f"Choose a GPT model", models)

        if  selected_model == "gpt-4-8k":
            st.session_state.gpt_model = 'gpt-4'
            token_limit = 8000
            prompt_cost = 0.03
            completion_cost = 0.06
        else:
            st.session_state.gpt_model = 'gpt-4-32k'
            token_limit = 32000
            prompt_cost = 0.06
            completion_cost = 0.12

    submit = st.sidebar.button("Translate")

    if 'translation_initial' not in st.session_state:
        st.session_state.translation_initial = ''

    if 'estimated_cost_prompt_initial' not in st.session_state:
        st.session_state.estimated_cost_prompt_initial = 0

    if 'estimated_cost_completion_initial' not in st.session_state:
        st.session_state.estimated_cost_completion_initial = 0

    if 'estimated_cost_total_initial' not in st.session_state:
        st.session_state.estimated_cost_total_initial = 0

    if submit:
        #########################################Initial Translation##########################################
        translation_initial = ""
        estimated_cost_prompt_initial = 0
        estimated_cost_completion_initial = 0
        estimated_cost_total_initial = 0
        i = 0
        for chunk in st.session_state.chunks:
            time.sleep(5)
            i =+ 1
            st.write(f"Translating Chunk {i}...")
            translation, prompt_token, completion_token = gpt_translator(
                st.session_state.gpt_model,
                chunk,
                st.session_state.user_prompt,
                # st.session_state.user_example,
                # st.session_state.assistant_example,
                st.session_state.temperature,
                token_limit
            )

            estimated_cost_prompt = prompt_token / 1000 * prompt_cost
            estimated_cost_completion = completion_token / 1000 * completion_cost
            estimated_cost_total = prompt_token / 1000 * prompt_cost + completion_token / 1000 * completion_cost

            translation_initial = translation_initial + '\n' + translation
            estimated_cost_prompt_initial += estimated_cost_prompt
            estimated_cost_completion_initial += estimated_cost_completion
            estimated_cost_total_initial += estimated_cost_total

        st.session_state.translation_initial = translation_initial
        st.session_state.estimated_cost_prompt_initial += estimated_cost_prompt_initial
        st.session_state.estimated_cost_completion_initial += estimated_cost_completion_initial
        st.session_state.estimated_cost_total_initial += estimated_cost_total_initial

        #################################### Refined Translation ###########################################
        # user_prompt_refined = ""
        # prompt_tokens = tiktoken_len(st.session_state.user_prompt)
        # example_user_tokens = tiktoken_len(st.session_state.user_example)
        # example_assistant_tokens = tiktoken_len(st.session_state.assistant_example)
        #
        # total_prompt_token = prompt_tokens + example_user_tokens + example_assistant_tokens
        #
        # gpt_token_limit = 8000
        #
        # token_room = gpt_token_limit * 0.9 - total_prompt_token
        #
        # if token_room > 0:
        #     gpt_model = 'gpt-4-8k'
        #     prompt_cost = 0.03
        #     completion_cost = 0.06
        # else:
        #     gpt_model = 'gpt-4-32k'
        #     gpt_token_limit = 32000
        #     token_room = gpt_token_limit * 0.9 - total_prompt_token
        #     prompt_cost = 0.06
        #     completion_cost = 0.12
        #
        # st.session_state.recommended_model = gpt_model
        #
        # chunk_token_limit = token_room / 4  # why am I dividing this by 4 (out of tokens left, 1 to the chunk size and 3 to the translated outcome)
        #
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_token_limit,
        #     chunk_overlap=0,  # number of tokens overlap between chunks
        #     length_function=tiktoken_len,
        #     separators=['\n\n', '\n', ' ', '']
        # )
        #
        # chunks = text_splitter.split_text(text)
        # st.session_state.chunks = chunks
        # st.write(f"Total Number of Chunks: {len(st.session.chunks)}")





    with col1:
        text_chunks = ['Text']
        # Prepare hover texts
        prompt_hover_texts = ['${:.2f}'.format(st.session_state.estimated_cost_prompt_initial)]
        completion_hover_texts = ['${:.2f}'.format(st.session_state.estimated_cost_completion_initial)]
        total_texts = ['${:.2f}'.format(st.session_state.estimated_cost_total_initial)]

        fig = go.Figure(data=[
            go.Bar(
                name='Prompt Cost',
                x=text_chunks,
                y=[st.session_state.estimated_cost_prompt_initial],
                orientation='v',
                hoverinfo='text',
                hovertext=prompt_hover_texts
            ),
            go.Bar(
                name='Completion Cost',
                x=text_chunks,
                y=[st.session_state.estimated_cost_completion_initial],
                orientation='v',
                hoverinfo='text',
                hovertext=completion_hover_texts,
                text=total_texts,  # Display value above the bar
                textposition='outside'
            )
        ])

        # Change the bar mode
        fig.update_layout(barmode='stack',
                          xaxis_title="Text",
                          yaxis_title="Estimated Cost (USD)")

        col1.markdown("### Translation Costs")
        col1.plotly_chart(fig)


    with col2:
    # Display bar graph based on translation_history

        col2.markdown("<br/>", unsafe_allow_html=True)
        col2.markdown("### Translation")
        col2.write(st.session_state.translation_initial)

    # Add Clear All button
    clear_all = st.sidebar.button("Clear All")

    # Handle the Clear All button click
    if clear_all:
        st.session_state.text_to_translate = ''
        st.session_state.user_prompt = ''
        st.session_state.user_example = ''
        st.session_state.assistant_example = ''
        st.session_state.azure_openai_api_base = ''
        st.session_state.azure_openai_api_version = ''
        st.session_state.azure_openai_api_key = ''
        st.session_state.translation_history = []
        st.session_state.translation_initial = ''
        st.session_state.estimated_cost_prompt_initial = 0
        st.session_state.estimated_cost_completion_initial = 0
        st.session_state.estimated_cost_total_initial = 0
        st.session_state.total_cost_estimate = 0
        st.experimental_rerun()


def main():
    translator()
    # # Add tabs
    # tabs = ["Cost Calculator", "Translator"]
    # selected_tab = st.selectbox("Choose a function", tabs)
    #
    # st.markdown("<br/>", unsafe_allow_html=True)
    # st.markdown("<br/>", unsafe_allow_html=True)
    # if selected_tab == "Cost Calculator":
    #     cost_calculator()
    # elif selected_tab == "Translator":
    #     translator()


if __name__ == "__main__":
    main()


