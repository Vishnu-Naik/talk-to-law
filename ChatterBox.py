
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter, 
    TokenTextSplitter
)
from langchain.vectorstores import Chroma


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from typing import Union


class ChatterBox:
    def __init__(self):
        self.documents = None
        self.doc_content = None
        self.splits = None
        self.vector_store_db = None

    def load_document(self, media_type, **kwargs):
        loader = None
        if media_type == 'web':
            web_url = kwargs.get('url')
            loader = WebBaseLoader(web_url)
        elif media_type == 'pdf':
            file_path = kwargs.get('path')
            loader = PyPDFLoader(file_path)
        
        if loader is None:
            raise ValueError('Invalid document type')
        self.documents = loader.load()
        self.doc_content = self.documents[0].page_content
    
    def split_document(self, split_type='recursive', chunk_size=500, chunk_overlap=20, doc_separator: Union[list, str] = []):
        if self.doc_content is None:
            raise ValueError('Document not loaded')

        splitter = None
        
        if split_type == 'recursive':
            doc_separator = doc_separator.extend(["\n\n", "\n", "\. ", " ", ""])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                # separators=["table of contentsSection", "\n\n", "\n", "\. ", " ", ""]
                separators=doc_separator
            )
        elif split_type == 'character':
            separator = doc_separator if doc_separator else "\n\n"
            splitter = CharacterTextSplitter(        
                # separator = "\n\n",
                # separator="/contentsSection\s[0-9][a-z]?/gm",
                # separator="table of contentsSection",	
                separator = separator,
                chunk_size = chunk_size,
                chunk_overlap  = chunk_overlap,
                length_function = len,
            )
        elif type == 'token':
            splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if splitter is None:
            raise ValueError('Invalid text splitter type')
            
        self.splits = splitter.split_text(self.doc_content)
        
    def get_vector_store_db(self, *, persist_directory: str='docs/chroma/', embeddings):
        croma_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        self.vector_store_db = croma_db.from_texts(self.splits, embedding=embeddings)
        return self.vector_store_db
    
    def semantic_search_query_on_db(self, query: str, num_results=10, compress=True, **kwargs):
        if not compress:
            return self.vector_store_db.max_marginal_relevance_search(query, k=num_results)
        else:
            if 'llm' not in kwargs:
                raise ValueError('LLM not provided for compression')
            llm = kwargs.get('llm')
            compressor = LLMChainExtractor.from_llm(llm)
            vector_store_db = self.get_vector_store_db(persist_directory='docs/chroma/', embeddings=llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vector_store_db.as_retriever(search_type = "mmr")
            )
            return compression_retriever.get_relevant_documents(query)
    
    @staticmethod
    def formulate_prompt():
        template = """HI, Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum. Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        return PromptTemplate(input_variables=["context", "question"],template=template,)
        
    def get_answer_for_query_and_context_from_llm(self, query: str, chat_history: str, llm):
        QA_CHAIN_PROMPT = self.formulate_prompt()
        result = None
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=self.vector_store_db.as_retriever(search_type = "mmr"),
                return_source_documents=True,
                # return_generated_question=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                # verbose=True
            )
            result = qa_chain({"query": query, "chat_history": chat_history})
            chat_history.extend([(query, result["result"])])
        except Exception as e:
            print("Error occured while operating the LLM chain")
        return result, chat_history


# def main():
    # web_url = "https://www.gesetze-im-internet.de/englisch_aufenthg/englisch_aufenthg.html"
    # chunk_size = 500
    # chunk_overlap = 20
    # embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    # # embeddings = OpenAIEmbeddings()
    
    # legal_bot = LegalBot()
    # legal_bot.load_document(media_type='web', url=web_url)
    # legal_bot.split_document(split_type='recursive',
    #                          chunk_size=chunk_size,
    #                          chunk_overlap=chunk_overlap,
    #                          doc_separator=["table of contentsSection"])
    # legal_bot.get_vector_store_db(persist_directory='docs/chroma/', embeddings=embeddings)
    
    # query = input("Hi, How can I help you with residence law?\n")
    
    # chat_history = []
    # # chat_llm = Cohere(cohere_api_key=cohere_api_key, temperature=0)
    # # chat_llm = OpenAI(temperature=0)
    # # chat_llm = AI21(ai21_api_key=ai21_api_key)
    # chat_llm = JinaChat()
    # while query != 'exit':
    #     result, chat_history = legal_bot.get_answer_for_query_and_context_from_llm(
    #         query=query, 
    #         chat_history=chat_history, 
    #         llm=chat_llm)
    #     print(result["result"]) if result is not None else print("Sorry for the inconvenience, Please try again!")
    #     query = input("\nDo you have any other questions? If not, please type 'exit' to end the conversation.\n")


# if __name__ == "__main__":
    
#     main()
    # import panel as pn
