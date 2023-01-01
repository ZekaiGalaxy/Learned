# Maria: A Visual Experience Powered Conversational Agent

### retriever
```
def hinge(x):
    return torch.clamp(x, min=0.)
# Hinge Loss
pos_loss = hinge(margin - true_pos_score + false_pos_score)
```

### vocab bias
```
#------add tag bias logits to response generation by Jokie---------
sequence_output_tag_part = sequence_output[tag_pos==1, :] #[b_s* tag_num, hidden_emb_size]
transformed_output_tag_part = self.transform(sequence_output_tag_part)
tag_logits_ori = self.decoder(transformed_output_tag_part) #[b_s* tag_num, vocab_size]
tag_logits = tag_logits_ori.mean(0) #[ vocab_size]

#mask the non-tag vocab part
tag_vocab_mask = torch.zeros(self.vocab_size, dtype=torch.float,device=tag_logits.device)
tag_vocab_mask[self.ids_for_tag_vacab] = 1.0

#add tag logit bias
tag_logits = tag_vocab_mask * tag_logits
class_logits = class_logits + tag_logits
```

### masked prediction
```
sequence_output_masked = sequence_output[masked_pos==1, :]
            
transformed_output_masked = self.transform(sequence_output_masked)
class_logits = self.decoder(transformed_output_masked)
```

### other
```
structure
- model.py
- loss.py
- metric.py
- main.py
- param.py (args/optimizer)
```
