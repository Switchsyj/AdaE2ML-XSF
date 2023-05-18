import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert_model/bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert_model/bert-base-uncased')
    
    text = '[CLS] book a table for [MASK] [MASK] next mar [SEP]'

    tokens = tokenizer.tokenize(text)
    token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    segment_ids = torch.tensor([[0] * len(tokens)]) 
    with torch.no_grad():
        predictions = model(token_ids, segment_ids)
        
    print(tokenizer.convert_ids_to_tokens(predictions[0][0].argmax(-1)[1:-1]))
