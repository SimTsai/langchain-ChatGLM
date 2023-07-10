# langchain-ChatGLM

esay use ChatGLM in LangChain.

[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
[langchain](https://github.com/hwchase17/langchain)

## Install Requirement
```
pip install -r requirements.txt
```

## Usage
```python

from chatglm_pipline import ChatGLMPipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = ChatGLMPipeline.from_model_id(
    model_id="THUDM/chatglm2-6b",
    device=-1, # if use GPU set to 0
    model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
    callback_manager=callback_manager, 
    verbose=True,
)

template = """问: {question}

答: 让我们一步一步地思考."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "华晨宇和张碧晨是什么关系?"

llm_chain.run(question)

```

## model_kwargs
| key | values | remark |
| --- | ------ | ------ |
| `"device"` | `"cuda"` | 使用cuda加速 |
| `"float"` | `True`, `False` | 使用cpu推理  |
| `"quantize"`| `8` | 量化 |

