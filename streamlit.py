if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content': prompt})

    # Prompt templates for short and detailed answers
    SHORT_ANSWER_TEMPLATE = """
    Answer the user's question briefly using only the relevant facts from the context. Avoid any extra explanations, examples, or metadata.
    If you don't know the answer, say "I don't know." Do not make up answers. Stick strictly to the context.

    Context: {context}
    Question: {question}

    Provide a short and precise answer.
    """

    DETAILED_ANSWER_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question. 
    If you don't know the answer, say "I don't know." Do not make up answers. Stick strictly to the context.

    Context: {context}
    Question: {question}

    Start the answer directly. Avoid small talk or unnecessary explanations.
    """

    #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
    #HF_TOKEN=os.environ.get("HUGGINGFACEHUB_API_TOKEN")  #local
    #HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"] # web
    
    # Pick prompt based on dropdown selection
    selected_template = SHORT_ANSWER_TEMPLATE if answer_type == "Short" else DETAILED_ANSWER_TEMPLATE

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")
            st.stop()
            #return

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=st.secrets["GROQQ_API_KEY"],
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(selected_template)}
        )

        response = qa_chain.invoke({'query': prompt})
        result = response["result"]
        source_documents = response["source_documents"]

        # Show result based on answer type
        if answer_type == "Short":
            result_to_show = result  # Just the concise answer
        else:
            # Append source info for detailed answers
            source_info = "\n".join([
                f"- Page {doc.metadata.get('page', '?')} from `{os.path.basename(doc.metadata.get('source', ''))}`"
                for doc in source_documents
            ])
            result_to_show = f"{result}\n\n**Source Documents:**\n{source_info}"

        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

    except Exception as e:
        st.error(f"Error: {str(e)}")
