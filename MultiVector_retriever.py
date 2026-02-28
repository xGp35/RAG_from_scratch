import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.stores import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


# Step 1: docs = load_documents()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
# A Document has 
# Document(
#     id: str
#     page_content: str,
#     metadata: dict
#     type: str
# )

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())

# Step 2:chunks = split(docs) -> No explicit chunking is happening here.
# I create summaries of full documents and use them as the searchable units

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-4o-mini",max_retries=0)
    | StrOutputParser()
)

# The variable summaries -> Kind of corresponds to chunks.
summaries = chain.batch(docs, {"max_concurrency": 5})

# Step 3: embeddings = embed(chunks)
# When I call "retriever.vectorstore.add_documents(summary_docs)" - That + this vector_store line is the embedding + storing
# Just the difference is here I first create the vectorstore, then add elements to it -> embedding happens during addition.
# The add_documents function does the embeddings
# Here I am not explicitly embedding like "Step 3" shows above, but instead using vectorstore-managed embeddings.     

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id" # This has to be part of the documents

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# Create unique ids for each document in the document chunk.
doc_ids = [str(uuid.uuid4()) for _ in docs]
# Now I have an doc_id attached to each document of the document list.
# I also know the doc_id to index mapping for each document.
# for i, doc_id in enumerate(doc_ids):
#   print(f"{i}: {doc_id}") 

# You defined:
# id_key = "doc_id"
# Then used it here:
# metadata={id_key: doc_ids[i]}
# So this becomes:
# metadata = {
#     "doc_id": "some-uuid"
# }

# This is completely custom.

# Here Lance is creating document objects for each summary generated.
summary_docs = []
for i,s in enumerate(summaries):
    document_obj = Document(page_content=s, metadata={id_key: doc_ids[i]})
    summary_docs.append(summaries)

# The code above is equivalent to the code below
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add
# Step 4: vector_store = store(embeddings) - The line just below this does this. 
retriever.vectorstore.add_documents(summary_docs)
# This line is not part of basic RAG. This is extra for storing Parent docs.
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Step 5: query_vec = embed(user_query)
# Step 6: contexts = vector_store.similarity_search(query_vec, k=3)
# Step 5 happens automatically inside, in the line marked as ø and π
# Step 6 happens in ø -> for child docs only
# Step 6 happens in π -> for child and parent

query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query,k=1)  # ø
# sub_docs[0] - is a Document object that is one of the summary_docs. this summary_doc matched the most with the user query.

retrieved_docs = retriever.invoke(query,n_results=1) # π
retrieved_docs[0].page_content[0:500]

# Step 7: prompt = build_prompt(contexts, user_query) -> Not present in the script
# Step 8: answer = llm(prompt) -> Also not present