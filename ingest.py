import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url="http://localhost:6333",
    prefer_grpc=False,
    collection_name="vector_database"
)

print("Vector DB Successfully Created!")