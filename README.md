# GraphRAG

> Step1. 在 root folder 建立 input 資料夾
```bash
mkdir -p ./input
```
> Step2. 將資料放入 input 資料夾

> Step3. 初始化 GraphRAG
```bash
python -m graphrag.index --init --root .
```
> Step4. (Optional) 修改 prompts，將其翻譯成繁體中文

> Step5. 修改 .env 設定 API KEY
```bash
GRAPHRAG_API_KEY=
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small
GRAPHRAG_LLM_MODEL=gpt-4o-mini
```
> Step6. 修改 settings.yaml 設定模型
```bash
model: ${GRAPHRAG_LLM_MODEL}
```
> Step7. 執行 Graph Indexing
```bash
python -m graphrag.index --root .
```
> Step8. 執行 Local Search
```bash
python -m graphrag.query --root . --method local "{query}"
```