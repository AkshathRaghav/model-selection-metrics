# Used in metrics::EMMS. Purpose is to handle diverse label formats by transforming them into a unified label embedding space using foundation models like CLIP, BERT, and GPT-2.

import torch 
from typing import List, Union 

def _embed_targets(targets: List[Union[str, int]], k: List[str] = ['clip']) -> torch.Tensor:
    def fix_tokenizer(tokenizer): 
        if tokenizer.pad_token_id is None: 
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else: 
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    stack = []
    if 'clip' in k: 
        from transformers import CLIPModel, CLIPProcessor
        clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
        clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
        inputs = clip_processor(text=targets, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        stack.append(outputs)
    if 'bert' in k: 
        from transformers import BertTokenizer, BertModel
        bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        fix_tokenizer(bert_tokenizer)
        bert_model = BertModel.from_pretrained('bert-large-uncased')
        inputs = bert_tokenizer(targets, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        stack.append(outputs.last_hidden_state.mean(dim = 1))
    if 'gpt-2' in k: 
        from transformers import GPT2Tokenizer, GPT2Model
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        fix_tokenizer(gpt2_tokenizer)
        gpt2_model = GPT2Model.from_pretrained('gpt2-medium')
        inputs = gpt2_tokenizer(targets, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
        stack.append(outputs.last_hidden_state.mean(dim = 1))

    return torch.stack(tuple(stack), axis=2)

def test_embed_targets(): 
    embeddings = _embed_targets(["hello world", "goodbye world"], k=["clip", "bert", "gpt-2"]) 
    print(embeddings)
    assert embeddings.shape == (2, 1024, 3)

test_embed_targets()