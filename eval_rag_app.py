import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from eval_metrics import * 

if 'api_key' not in st.session_state: st.session_state['api_key'] = None
if 'user_turn' not in st.session_state: st.session_state['user_turn'] = False
if 'pdf' not in st.session_state: st.session_state['pdf'] = None
if "embed_model" not in st.session_state: st.session_state['embed_model'] = None
if "eval_models" not in st.session_state: st.session_state["eval_models"] = {"app_metrics": AppMetrics()}

st.set_page_config(page_title="Document Genie", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@AppMetrics.measure_execution_time
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=st.session_state['api_key'])
    vector_store = FAISS.from_texts(text_chunks, embedding=st.session_state['embed_model'], 
                                    distance_strategy=DistanceStrategy.DOT_PRODUCT)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I don't think the answer is available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=st.session_state['api_key'])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@AppMetrics.measure_execution_time
def llm_output(chain, docs, user_question):
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)


@AppMetrics.measure_execution_time
def fetch_context(new_db, user_question, k = 3, distance_strategy=DistanceStrategy.DOT_PRODUCT):
    return new_db.similarity_search_with_score(user_question, k = 3, distance_strategy=DistanceStrategy.DOT_PRODUCT)

@st.cache_data
def user_input(user_question):
    # embeddings =  OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=st.session_state['api_key'])
    new_db = FAISS.load_local("faiss_index", st.session_state['embed_model'], 
                              allow_dangerous_deserialization=True, 
                              distance_strategy=DistanceStrategy.DOT_PRODUCT)
    
    contexts_with_scores, exec_time = fetch_context(new_db, user_question, k = 3, distance_strategy=DistanceStrategy.DOT_PRODUCT)
    st.session_state["eval_models"]["app_metrics"].exec_times["chunk_fetch_time"] = exec_time

    docs = [i[0] for i in contexts_with_scores]
    chain = get_conversational_chain()
    response, exec_time = llm_output(chain, docs, user_question)
    st.session_state["eval_models"]["app_metrics"].exec_times["llm_resp_time"] = exec_time
    
    st.write("Reply: ", response["output_text"])

    ctx = ""
    for (item, score) in contexts_with_scores:
        if len(item.page_content.strip()):
            ctx += f"<li>Similarity Score: {round(float(score), 2)}<br>Context: {item.page_content}<br>&nbsp</li>"

    with st.expander("Click to see the context passed"):
        st.markdown(f"""<ol>{ctx}</ol>""", unsafe_allow_html=True)
    
    return contexts_with_scores, response["output_text"]


def evaluate_all(query, context_lis, response):
    guard =  st.session_state["eval_models"]["guards"]
    stat =  st.session_state["eval_models"]["textstat"]
    comp =  st.session_state["eval_models"]["comparison"]
    context = "\n\n".join(context_lis) if len(context_lis) else "no context"
    
    RESULT = {}

    RESULT["guards"] = {
        "query_injection": guard.prompt_injection_classif(query),
        "context_injection": guard.prompt_injection_classif(context),
        "query_bias": guard.bias(query),
        "context_bias": guard.bias(context),
        "response_bias": guard.bias(response),
        "query_regex": guard.detect_pattern(query),
        "context_regex": guard.detect_pattern(context),
        "response_regex": guard.detect_pattern(response),
        "query_toxicity": guard.toxicity(query),
        "context_toxicity": guard.toxicity(context),
        "response_toxicity":  guard.toxicity(response),
        "query_sentiment": guard.sentiment(query),
        "query_polarity": guard.polarity(query),
        "context_polarity":guard.polarity(context), 
        "response_polarity":guard.polarity(response), 
        "query_response_hallucination" : comp.hallucinations(query, response),
        "context_response_hallucination" : comp.hallucinations(context, response),
        "query_response_hallucination" : comp.contradiction(query, response),
        "context_response_hallucination" : comp.contradiction(context, response),
    }

    RESULT["guards"].update(guard.harmful_refusal_guards(query, context, response))

    tmp = {}
    for key, val in comp.ref_focussed_metrics(query, response).items():
        tmp[f"query_response_{key}"] = val

    for key, val in comp.ref_focussed_metrics(context, response).items():
        tmp[f"context_response_{key}"] = val
    
    RESULT["reference_based_metrics"] = tmp
    
    
    tmp = {}
    for key, val in comp.string_similarity(query, response).items():
        tmp[f"query_response_{key}"] = val

    for key, val in comp.string_similarity(context, response).items():
        tmp[f"context_response_{key}"] = val
    
    RESULT["string_similarities"] = tmp

    tmp = {}
    for key, val in stat.calculate_text_stat(response).items():
        tmp[f"result_{key}"] = val
    RESULT["response_text_stats"] = tmp

    RESULT["execution_times"] = (st.session_state["eval_models"]["app_metrics"].exec_times)
    
    return RESULT


def main():
    st.markdown("""## RAG Pipeline Example""")
    
    st.info("Note: This is a minimal demo focussing on ***EVALUATION*** so you can do simple Document QA which uses GPT-3.5 without any persistant memory hence no multi-turn chat is available there. If the question is out of context from the document, this will not work so ask the questions related to the document only. You can optimise the workflow by using Re-Rankers, Chunking Strategy, Better models etc but this app runs on CPU right now easily and is about, again, ***EVALUATION***", icon = "ℹ️")

    st.error("WARNING: If you reload the page, everything (model, PDF, key) will have to be loaded again. That's how `streamlit` works", icon = "🚨")
    
    with st.sidebar:
        st.title("Menu:")
        st.session_state['api_key'] = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
        st.session_state["pdf"] = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
       
        if st.session_state["pdf"]:
            if st.session_state["embed_model"] is None: 
                with st.spinner("Setting up `all-MiniLM-L6-v2` for the first time"):
                    st.session_state["embed_model"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


            with st.spinner("Processing PDF files into chunks and creating `FAISS` Index..."):
                raw_text = get_pdf_text(st.session_state["pdf"])
                text_chunks, exec_time = get_text_chunks(raw_text)

                st.session_state["eval_models"]["app_metrics"].exec_times["chunk_creation_time"] = exec_time

                get_vector_store(text_chunks)
                st.success("Done")


    if not st.session_state['api_key']: st.warning("Enter OpenAI API Key to proceed")
    elif not st.session_state["pdf"]: st.warning("Upload a PDf file")
    else:
        st.markdown("""#### Ask a Question from the PDF file""")
        user_question = st.text_input("", key="user_question")

        if user_question and st.session_state['api_key']:  # Ensure API key and user question are provided
            contexts_with_scores, response = user_input(user_question)
        
            st.warning("There are 5 major types metrics computed below having multiple sub metrics. Also, 2 abstract classes are defined `LLMasEvaluator` (to use any LLM as a judge) and `TraditionalPipelines` (for Topics, NER, POS etc)", icon="🤖")
            metric_calc = st.button("Load Models & Compute Evaluation Metrics")
            if metric_calc:
                if len(st.session_state["eval_models"]) <= 1:
                    st.session_state["eval_models"].update({
                        "guards": IOGuards(),
                        "textstat": TextStat(),
                        "comparison": ComparisonMetrics(),
                        # "llm_eval": LLMasEvaluator(),
                        # "traditional_pipeline": TraditionalPipelines(),
                        })

                with st.spinner("Calculating all the matrices. Please wait ...."):
                    eval_result = evaluate_all(user_question, [item.page_content for (item, score) in contexts_with_scores], response)
                    st.balloons()

                # with st.expander("Click to see all the evaluation metrics"):
                    st.json(eval_result)


if __name__ == "__main__":
    main()