from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFDocumentProcessor:
    def __init__(self, file_path, chunk_size=500, chunk_overlap=20, title=""):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.title = title

    def load_and_split(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        docs = text_splitter.split_documents(pages)
        docs_with_metadata = [
            [d.page_content, self.title] for d in docs
        ]

        return docs_with_metadata
