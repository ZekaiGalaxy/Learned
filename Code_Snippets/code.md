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
# Lite Unified Modeling for Discriminative Reading Comprehension

### POS embedding
```
self.pos_embeddings = nn.Embedding(40, config.embedding_size)
if pos_tags is not None:
    pos_embeddings = self.pos_embeddings(pos_tags)
    embeddings = inputs_embeds + position_embeddings + token_type_embeddings + pos_embeddings
```

### CoAttention
```
def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    return result
    
def seperate_seq(sequence_output, con_start, con_end):
    que_seq_output = sequence_output.new(sequence_output.size()).zero_()
    con_seq_output = sequence_output.new(sequence_output.size()).zero_()
    que_attention_mask = sequence_output.new_zeros(sequence_output.size(0), sequence_output.size(1))
    con_attention_mask = sequence_output.new_zeros(sequence_output.size(0), sequence_output.size(1))
    for i in range(sequence_output.size(0)):
        que_seq_output[i, :con_start[i]] = sequence_output[i, :con_start[i]]
        con_seq_output[i, con_start[i]:con_end[i]+1] = sequence_output[i, con_start[i]:con_end[i]+1]
        que_attention_mask[i, :con_start[i]] = sequence_output.new_ones(sequence_output.size(1))[:con_start[i]]
        con_attention_mask[i, con_start[i]:con_end[i]+1] = sequence_output.new_ones(sequence_output.size(1))[con_start[i]:con_end[i]+1]
    return que_seq_output, con_seq_output, que_attention_mask, con_attention_mask


    
context_seq_output, ending_seq_output, context_attention_mask, ending_attention_mask = seperate_seq(
            sequence_output, con_start, con_end)
context_embedding = context_seq_output.max(1)[0].unsqueeze(1).repeat(1, sequence_output.size(1), 1)
ending_embedding = ending_seq_output.max(1)[0].unsqueeze(1).repeat(1, sequence_output.size(1), 1)
context_scores = torch.cosine_similarity(sequence_output, ending_embedding, dim=-1)
ending_scores = torch.cosine_similarity(sequence_output, context_embedding, dim=-1)
context_scores = (context_scores + 1) / 2
ending_scores = (ending_scores + 1) / 2
```

### framework to learn
```
huggingface : run_glue, run_multiple_choice.py ...
fairseq
```

# Synthetic Question Value Estimationfor Domain Adaptation of Question Answering

### RL
```
qa_model.zero_grad()
qve_model.zero_grad()
for epoch in range(EPOCH):
    qve_model.train()
    # select 
    select_prob = Binomial(1, qa_values).sample()
    # train qa
    qa_loss = train_qa()
    # RL
    reward = get_reward()
    prob = select_prob * torch.log(qa_values + epsilon) + (1 - select_prob) * torch.log(1 - qa_values + epsilon)
    qve_loss_rl = -reward * prob.mean()
    loss.backward()
```
