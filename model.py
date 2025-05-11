from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

device = torch.device("mps" if torch.mps.is_available() else "cpu")

model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
state_dict = torch.load("best_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.save_pretrained("./fake_news_model")

tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
tokenizer.save_pretrained("./fake_news_model")




