import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template_text):
    return PromptTemplate(template=template_text, input_variables=["context", "question"])


def load_llm(provider, prompt_template):
    if provider == "Groq":
        return ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            groq_api_key=st.secrets["GROQQ_API_KEY"]
        )
    else:
        return HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  # Free HuggingFace model
            temperature=0.5,
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            #model_kwargs={"max_length": 512}
        )


def main():
    st.title("Ask Chatbot!")

    # User selections
    provider = st.selectbox("Select LLM Provider", ["Groq", "HuggingFace"])
    answer_type = st.selectbox("Select Answer Type", ["Short", "Detailed"])

    # Prompt templates
    SHORT_ANSWER_TEMPLATE = """
    Answer the user's question briefly using only the relevant facts from the context. Avoid any extra explanations, examples, or metadata.
    If you don't know the answer, say "I don't know." Do not make up answers. Stick strictly to the context.

    Context: {context}
    Question: {question}

    Avoid small talk or unnecessary explanations. Provide precise answer.
    """

    DETAILED_ANSWER_TEMPLATE = """
    Use the pieces of information provided in the context to answer the user's question. 
    If you don't know the answer, say "I don't know." Do not make up answers. Stick strictly to the context.

    Context: {context}
    Question: {question}

    Start the answer directly. Avoid small talk or unnecessary explanations.
    """

    # Select appropriate prompt
    selected_template = SHORT_ANSWER_TEMPLATE if answer_type == "Short" else DETAILED_ANSWER_TEMPLATE

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            llm = load_llm(provider, selected_template)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(selected_template)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_docs = response["source_documents"]

            if answer_type == "Detailed":
                source_info = "\n".join([
                    f"- Page {doc.metadata.get('page', '?')} from `{os.path.basename(doc.metadata.get('source', ''))}`"
                    for doc in source_docs
                ])
                result_to_show = f"{result}\n\n**Source Documents:**\n{source_info}"
            else:
                result_to_show = result

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
