from haystack.nodes import PromptNode, PromptTemplate
from haystack.schema import Document

summarize = PromptTemplate(
    name="summarize",
    prompt_text="将以下内容使用中文进行总结: $doc, 总结：",
)

QA = PromptTemplate(
    name="QA",
    prompt_text="Document: $doc; Question: $question; Answer:",
)

prompt_node = PromptNode(model_name_or_path="text-davinci-003",api_key='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')

with open('./result.txt', 'w', encoding='utf-8') as f:
    f.writelines(prompt_node.prompt(QA, doc="peter is a boy who was borned in 1973. Today is 2023-2-26.", question="how old is peter?"))