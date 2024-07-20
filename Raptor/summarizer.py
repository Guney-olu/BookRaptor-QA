"""
This Class has different ways to summarize the text
[open-source, openai api, langchain+openai]
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain_openai import OpenAI

class TextSummarizer:
    def __init__(self, openai_api_key=None, openai_org_key=None):
        self.openai_api_key = openai_api_key
        self.openai_org_key = openai_org_key

    # OPENSOURCE/FREE
    def t5_summary(self, text):
        tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-base")
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_new_tokens=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # GPU Needed take 22 GB [use kaggle]
    # TODO: add quantize model and tinyops support (https://github.com/Guney-olu/tinyOPS)
    def llama_summary(self, text):
        model_id = "SalmanFaroz/Llama-2-7b-samsum" # special model for summary 
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        prompt = f"""You are an assistant to create a detailed summary of the text input provided.
                Text:
                {text}
                """
        inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        summary_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # TODO: key as env var for security and change this lame method of applying API key
    # PAID API
    def openai_summarize(self):
        if not self.openai_api_key or not self.openai_org_key:
            raise ValueError("OpenAI API key and organization key must be provided")
        
        openai_client = OpenAI(
            openai_api_key=self.openai_api_key, 
            openai_organization=self.openai_org_key,
            model_name="gpt-3.5-turbo-instruct"
        )
        return openai_client

# Example usage:
# summarizer = TextSummarizer(openai_api_key='your_api_key', openai_org_key='your_org_key')
# text = "Your text to summarize here"
# print(summarizer.t5_summary(text))
# print(summarizer.llama_summary(text))
# print(summarizer.openai_summarize())
