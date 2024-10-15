from langchain_groq import ChatGroq
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from few_shots import few_shots
def get_answer(question):

   llm = ChatGroq(
    groq_api_key='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',
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0.0,
    max_retries=2,)
   db_user = "root"
   db_password = "7639nivash10"
   db_host = "localhost"
   db_name = "atliq_tshirts"

   db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                             sample_rows_in_table_info=3)
   dbinfo = db.table_info
   embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

   to_vectorize = ["".join(str(value) for value in example.values() if not isinstance(value, dict)) for example in
                   few_shots]
   vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
   example_selector = SemanticSimilarityExampleSelector(
       vectorstore=vectorstore,
       k=2,
   )
   example_prompt = PromptTemplate(
       input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
       template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
   )
   few_shot_prompt = FewShotPromptTemplate(
       example_selector=example_selector,
       example_prompt=example_prompt,
       prefix=_mysql_prompt,
       suffix=PROMPT_SUFFIX,
       input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
   )
   new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
   response = new_chain(question)

if __name__ == '__main__':
    re = get_answer('how many tshirts do we have left in black color ')
    print(re)