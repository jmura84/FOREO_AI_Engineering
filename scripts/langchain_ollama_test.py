import time
from langchain_ollama.llms import OllamaLLM

# https://python.langchain.com/docs/integrations/llms/ollama/

def test_langchain_ollama(model_name: str, sentence: str) -> tuple:
    '''
    Testing LangChain integration with Ollama for English to Spanish translation.

    :param model_name: str
    :param sentence: str
    :return: tuple (translation, latency)
    '''
    # Base Prompt:
    prompt = (
        f'''
        You are a professional English to Spanish translator.
        Translate ONLY the following sentence into Spanish. Do NOT add explanations.
        Sentence: {sentence}
        Translation:
        '''
    )

    # Sampling parameters:
    llm = OllamaLLM(
        model=model_name,
        temperature=0.7,   # randomness
        top_p=0.9,         # nucleus sampling
        top_k=50,          # truncation
        num_predict=100    # maximum tokens
    )

    # Measuring starting time:
    start_time = time.perf_counter()

    # Calling the model:
    response = llm.invoke(prompt)

    # Measuring finishing time:
    end_time = time.perf_counter()
    latency = end_time - start_time

    # Return translation and latency:
    return response, latency


if __name__ == "__main__":
    # NOTE: The first try when switching models tends to be slower
    # because Ollama has to load the model. Run twice for more accurate latency.
    model_name = "gemma3:4b"
    # model_name = "gemma3:12b"
    # model_name = "gpt-oss:20b"
    sentence = "I want to become an AI Engineer and create SkyNet."

    translation, latency = test_langchain_ollama(model_name, sentence)

    print(f"Translation: {translation}")
    print(f"Latency: {latency:.2f} seconds")