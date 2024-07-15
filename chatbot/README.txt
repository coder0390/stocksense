pip install --upgrade-strategy eager optimum[openvino,nncf]
pip install -U accelerate peft transformers einops datasets bitsandbytes
pip install -U "huggingface_hub[cli]"

huggingface-cli login
token：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

To execute the chatbot, run all the cells of "chatbotfine2.ipynb"