"""
This file has diffrent ways to summarize the text
[open-source , openai api, langchain+openai]
"""

# OPENSOURCE/FREE
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM

def t5_summary(text):
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_new_tokens=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# GPU Needed take 22 gb [use kaggle]
#TODO add quantize modela and tinyops support (https://github.com/Guney-olu/tinyOPS)
def llama_summary(text):
    model_id =  "SalmanFaroz/Llama-2-7b-samsum" #special model for summary 
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompt = f"""You are an assistant to create a detailed summary of the text input prodived.
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



#TODO key as env var for security and change this lame method of aaplying api key
# PAID API - >> Add the keys here ... 

from langchain_openai import OpenAI


def openai_summarize(api_key,org_key):

    return OpenAI(openai_api_key=api_key, openai_organization=org_key,model_name="gpt-3.5-turbo-instruct")


# from openai import OpenAI
# client = OpenAI(
#   organization='YOUR_ORG_ID',
#   project='$PROJECT_ID',
# )
# def summary_openai(
#     text
# ):
#     res = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=
#        [
#          {
#           "role": "system",
#           "content": "You are a helpful assistant for text summarization.",
#          },
#          {
#           "role": "user",
#           "content": f"Summarize this {text}",
#          },
#         ],
#     )
#     for chunk in res:
#         if chunk.choices[0].delta.content is not None:
#             res = chunk.choices[0].delta.content
#     return res    