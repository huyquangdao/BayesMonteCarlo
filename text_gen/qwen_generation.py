from text_gen.llama3_generation import Llama3ConfigForGeneration, Llama3Generation

class QwenConfigForGeneration(Llama3ConfigForGeneration):
    # the prompt used for
    prompt = "This is the prompt and subjected to be changed"
    temperature = 0.6
    model_type = "qwen"
    pass

class QwenGeneration(Llama3Generation):
    pass