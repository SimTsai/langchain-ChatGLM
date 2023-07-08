"""Wrapper around ChatGLMP Pipeline APIs."""

from functools import partial
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

DEFAULT_MODEL_ID = "THUDM/chatglm2-6b"

logger = logging.getLogger(__name__)

class ChatGLMPipeline(LLM):
    """Wrapper around ChatGLM Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import ChatGLM
            hf = ChatGLM.from_model_id(
                model_id="THUDM/chatglm2-6b",
                model_kwargs={"trust_remote_code": True, device='cuda'}
            )
    """

    client: Any  #: :meta private:
    stream_client: Any #: :meta private:
    tokenizer: Any #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModel,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
           model = AutoModel.from_pretrained(model_id, **_model_kwargs)
           model = model.eval()
        except ImportError as e:
            raise ValueError(
                f"Could not load the {model_id} model due to missing dependencies."
            ) from e

        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        return cls(
            client=model.chat,
            stream_client=model.stream_chat,
            model_id=model_id,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs
        }

    @property
    def _llm_type(self) -> str:
        return "chatglm_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            return self.stream(prompt=prompt, stop=stop, run_manager=run_manager)
        else:
            response, history = self.client(self.tokenizer, prompt, history=[])
            return response

    def stream(
        self,
        prompt,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        current_length = 0
        text_callback = None
        if run_manager:
                text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        for response, history, past_key_values in self.stream_client(self.tokenizer, prompt, history=[],return_past_key_values=True):
            if text_callback:
                text_callback(response[current_length:])
            text += response[current_length:]
            current_length = len(response)
        return text
