import os

import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# ################################################################################################
# GraphRAG
# ################################################################################################
OUTPUT_DIR = "output/20240721-002740/artifacts"
LANCEDB_URI = f"{OUTPUT_DIR}/lancedb"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

################ Read entities ################
# read nodes table to get community and degree data
entity_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

# load description embeddings to an in-memory lancedb vectorstore
# to connect to a remote db, specify url and port values.
description_embedding_store = LanceDBVectorStore(
    collection_name="entity_description_embeddings",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)
entity_description_embeddings = store_entity_semantic_embeddings(
    entities=entities, vectorstore=description_embedding_store
)

################ Read relationships ################
relationship_df = pd.read_parquet(f"{OUTPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

################ Read community reports ################
report_df = pd.read_parquet(f"{OUTPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

################ Read text units ################
text_unit_df = pd.read_parquet(f"{OUTPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

################ Model setting ################
api_key = os.environ["GRAPHRAG_API_KEY"]
llm_model = os.environ["GRAPHRAG_LLM_MODEL"]
embedding_model = os.environ["GRAPHRAG_EMBEDDING_MODEL"]

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_base=None,
    api_type=OpenaiApiType.OpenAI,
    model=embedding_model,
    deployment_name=embedding_model,
    max_retries=20,
)

################ reate local search context builder ################
context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    # covariates=covariates,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
}

llm_params = {
    "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
    "temperature": 0.0,
}

LOCAL_SEARCH_SYSTEM_PROMPT = """
---角色---
你是一個有幫助的助理，負責回答關於提供的表格數據的問題。
---目標---
生成一個符合目標長度和格式的回應，以回答用戶的問題，根據回應的長度和格式適當地總結輸入數據表中的所有信息，並納入任何相關的一般知識。
如果你不知道答案，就直說。不要編造任何內容。
由數據支持的觀點應列出其數據參考，如下所示：
"這是一個由多個數據參考支持的示例句子 [數據：<數據集名稱> (記錄 ID)；<數據集名稱> (記錄 ID)]。"
在單一參考中不要列出超過 5 個記錄 ID。相反，列出最相關的前 5 個記錄 ID，並添加"+更多"以表示還有更多。
例如：
"X 先生是 Y 公司的所有者，並受到多項不當行為指控 [數據：來源 (15, 16)，報告 (1)，實體 (5, 7)；關係 (23)；聲明 (2, 7, 34, 46, 64, +更多)]。"
其中 15、16、1、5、7、23、2、7、34、46 和 64 代表相關數據記錄的 ID（而不是索引）。
不要包含沒有提供支持證據的信息。
---目標回應長度和格式---
{response_type}
---數據表---
{context_data}
---目標回應長度和格式---
{response_type}
- 根據長度和格式適當地為回應添加列點或步驟。
- 使用 Markdown 格式設計回應的樣式，但不使用 header # 符號。
- 不需添加數據來源。
- 使用台灣繁體中文回覆。
"""

search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
)

# result = await search_engine.asearch("Welly信心指數標準？")
# print(result.response)

# ################################################################################################
# Streamlit UI
# ################################################################################################
import streamlit as st
import asyncio

# credentials
user_credentials_dict = st.secrets["credentials"]

st.title("Software Engineering at Google")

# 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# 定義預設問題
default_questions = [
    "What is the difference between software engineering and programming according to the book?",
    "How does Google manage the sustainability and scalability of its massive codebase according to the book?",
    "What are some key practices Google follows to ensure effective software engineering, as mentioned in the book?",
    "Can you explain the concept of 'Hyrum's Law' as described in the book?",
    "What are the impacts of time and change on software engineering, according to Google's experiences shared in the book?",
    "How does Google handle large-scale changes and deprecations within its codebase as detailed in the book?",
    "Discuss Google’s philosophy on the costs and trade-offs in software engineering decisions as presented in the book.",
    "What role does culture play in Google’s software engineering practices as described in the book?",
]

# 創建兩行按鈕
button_pressed = ""

# 第一行按鈕
cols1 = st.columns(4)
for i in range(4):
    with cols1[i]:
        if st.button(default_questions[i], key=f"btn1_{i}"):
            button_pressed = default_questions[i]

# 第二行按鈕
cols2 = st.columns(4)
for i in range(4):
    with cols2[i]:
        if st.button(default_questions[i+4], key=f"btn2_{i}"):
            button_pressed = default_questions[i+4]

# 顯示聊天歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 獲取用戶輸入
if prompt := (st.chat_input("您需要查找什麼資料？") or button_pressed):
    # 添加用戶消息到聊天歷史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 顯示 AI 思考中的消息
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        inputs = {"question": prompt}

        # 添加 spinner
        with st.spinner('RAG Processing...'):            
            # 使用asyncio運行異步搜索
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(search_engine.asearch(prompt))
            response = result.response
            print(response)
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            