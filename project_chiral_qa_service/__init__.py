from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.schema import Document 
from haystack.nodes import OpenAIAnswerGenerator, DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

store = FAISSDocumentStore(
    faiss_index_factory_str="Flat",
    return_embedding=True,
)

MODEL_PATH = '/root/autodl-tmp/models'
retriever = DensePassageRetriever(
    document_store=store,
    query_embedding_model=f'{MODEL_PATH}/dpr-question_encoder-single-nq-base',
    passage_embedding_model=f"{MODEL_PATH}/dpr-ctx_encoder-single-nq-base",
    use_gpu=True,
    embed_title=True,
)
store.delete_documents()
store.write_documents([
    Document(content="peter is a boy who was borned in 1973.", meta={'author': 'observer'})
])
store.update_embeddings(retriever=retriever)

generator = OpenAIAnswerGenerator(model='text-davinci-003', api_key='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')

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

result = pipe.run(query="If today is 2023-02-16, How old is peter now?", params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
print_answers(result, details="minimum")