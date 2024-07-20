"""
This helper Func help to save data in milvus db
"""
import json
import tqdm
import ast
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

class MilvusDataHandler:
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_st = SentenceTransformer('all-MiniLM-L6-v2')
        self.milvus_client = MilvusClient(uri=self.db_path)
    
    def clean_text(self, text):
        text = text.replace('\t', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\u2019', "'")
        return text

    def prepare_data(self, parsed_data):
        data = []
        for i, entry in enumerate(tqdm.tqdm(parsed_data, desc="Preparing data")):
            combined_text = self.clean_text(entry['text'])
            embedding = self.model_st.encode(combined_text).tolist()  
            data.append({
                "id": i,
                "vector": embedding,
                "text": combined_text,
                "title": entry['title']
            })
        return data

    def parse_input_file(self, input_file):
        parsed_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                try:
                    entry = ast.literal_eval(line.strip())
                    
                    if isinstance(entry, list) and len(entry) == 2:
                        text = self.clean_text(entry[0].strip())
                        title = entry[1].strip()
                        
                        parsed_data.append({
                            'text': text,
                            'title': title
                        })
                    else:
                        print(f"Warning: Skipping line due to unexpected format: {line}")

                except Exception as e:
                    print(f"Error parsing line: {line}. Error: {e}")
        return parsed_data

    def save_data_to_json(self, data, json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def create_collection(self, dimension):
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            metric_type="IP",
            consistency_level="Strong", 
        )

    def insert_data(self, data):
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def run(self, input_file, json_file):
        parsed_data = self.parse_input_file(input_file)
        self.save_data_to_json(parsed_data, json_file)
        data = self.prepare_data(parsed_data)
        self.create_collection(dimension=len(data[0]['vector']))
        self.insert_data(data)
        print("Data insertion completed.")
