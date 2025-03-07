from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_dir = "./local_model"  # Specify your desired directory

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=model_dir)
