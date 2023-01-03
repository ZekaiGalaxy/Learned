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

# Persona-Guided Planning for Controlling the Protagonist’s Persona in Story Generation

### extract keywords
```
stop = stopwords.words('english') + list(string.punctuation)  # + ["'s", "'m", "'re", "'ve"]

sid = SentimentIntensityAnalyzer()
stemmer = WordNetLemmatizer()

def extract_emotion_event_at_least_one(text, limit=False, per_sent=False):
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(word, nltk_tag):
        tag = get_wordnet_pos(nltk_tag)
        return stemmer.lemmatize(word, tag).lower()

    def sample_sorted(words, num):
        ids = list(range(len(words)))
        choosed = random.sample(ids, num)
        choosed = sorted(choosed)
        res = []
        for i in choosed:
            res.append(words[i])
        return res

    def extract_emotion_event(sent, last_word):
        global sid, discard_words

        words = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(words)
        res = []
        res.append(last_word)
            

        for word, tag in tagged:
            if not word[0].isalpha():
                continue
            origin_word = lemmatize(word, tag)

            ss = sid.polarity_scores(word)
            for key in ss:
                if not word[0].isalpha():
                    continue
                if ss[key] > 0.5 and (key == 'pos' or key == 'neg'):
                   
                    res.append(origin_word)

            if get_wordnet_pos(tag) in [wordnet.VERB, wordnet.NOUN]:
                if origin_word in stop or word in stop:
                    continue
                if origin_word in discard_words or word in discard_words:
                    continue
                res.append(origin_word)

        res.pop(0)
        from math import ceil
        max_len = min(5, ceil(len(words) * 0.1))
        if limit and max_len < len(res):
            return sample_sorted(res, max_len)
        else:
            return res
```

### Knowledge Graph
```
class RelationList:
    def __init__(self):
        self.relation2id = {}
        self.cnt = 0
    
    def add(self, relation):
        if relation not in self.relation2id:
            self.relation2id[relation] = self.cnt
            self.cnt += 1
        return self.relation2id[relation]

    def get_idx(self, relation):
        return self.relation2id[relation]

    def __len__(self):
        return self.cnt
    
    def __repr__(self):
        result = ''
        for k, v in self.relation2id.items():
            result += f'{k} : {v}\n'
        return result


class KnowledgeGraph:

    def __init__(self, edges, bidir=False):
        self.data = {}
        self.prev = {}
        self.weights = {}
        self.relations = RelationList()

        self.relations.add('unrelated') # relation_id 0 对应和NOT_A_FACT的连接

        for item in edges:
            # [head, relation, tail, weight]
            head = item[0]
            relation = item[1]
            tail = item[2]

            if not self.eng_word(head) or not self.eng_word(tail):
                continue
            assert '/' not in head and '/' not in tail

            relation_id = self.relations.add(relation)
            self.add(head, relation_id, tail)
            if bidir:
                self.add(tail, relation_id, head)
            self.weights[self.get_name(item[0], item[-2])] = float(item[-1])
            if bidir:
                self.weights[self.get_name(item[-2], item[0])] = float(item[-1])

        print(f"relation nums:{len(self.relations)}")
        print(self.relations)
    
    def get_relation_size(self):
        # 返回relation总数
        return len(self.relations)
        
    def get_relation_list(self):
        return self.relations

    # def get_relation_idx(self, relation):
    # 加进图谱的时候已经转换成id了
    #     return self.relations.get_idx(relation)

    def filter_points(self, points):
        res = []
        for pt in points:
            if pt in self.data:
                res.append(pt)
        return res

    def check(self, point):
        return point in self.data

    def get_name(self, src, dst):
        return src + "___" + dst

    def get_weight(self, src, dst):
        name = self.get_name(src, dst)
        if name in self.weights:
            return self.weights[name]
        return None

    def eng_word(self, word):
        if '_' in word:
            return False
        return True

    def get_avg_deg(self):
        r = 0
        for src in self.data:
            r += len(self.data[src])

        return r / len(self.data)

    def show_degs(self):
        data = list(self.data.items())
        print(data[-3:])
        data.sort(key=lambda x: len(x[1]))
        for k, v in data:
            print(f'{k}:{len(v)}')

    def get_node_num(self):
        return len(self.data)

    def add(self, src, relation, dst):
        w = (dst, relation)
        if src in self.data:
            if w not in self.data[src]:
                self.data[src].append(w)
        else:
            self.data[src] = [w]

        q = (src, relation)
        if dst in self.prev:
            if q not in self.prev[dst]:
                self.prev[dst].append(q)
        else:
            self.prev[dst] = [q]


    def get_neighbors(self, pt, relation=False):
        if pt not in self.data:
            return []
        else:
            if relation:
                return self.data[pt]
            else:
                return [i[0] for i in self.data[pt]]


    def get_triples(self, word):

        res = []
        if word in self.data:
            for dst, r in self.data[word]:
                res.append((word, r, dst))

        if word in self.prev:
            for src, r in self.prev[word]:
                res.append((src, r, word))
        
        if not res:
            res.append((word, 0, 'NOT_A_FACT'))

        return res

    def get_hops_set(self, srcs, hop, relation=False):
        res = set(srcs)
        step = 0
        temp = set(srcs)
        while step < hop:
            step += 1
            new_temp = []
            for pt in temp:
                ns = self.get_neighbors(pt, relation=relation)
                for n in ns:
                    if n not in res:
                        new_temp.append(n)
            new_temp = set(new_temp)
            temp = new_temp
            res = res | new_temp
        return res
        
    def get_intersect(self, srcs, dsts, hop=2):
        src_neis = self.get_hops_set(srcs, hop)
        dst_neis = self.get_hops_set(dsts, hop)
        return src_neis & dst_neis

    def find_neigh_in_set(self, src, points):
        res = []
        if src not in self.data:
            return res
        for pt in points:
            if pt in self.data[src]:
                res.append(pt)
        return set(res)

    def find_paths(self, srcs, dsts):
        a = self.get_hops_set(srcs, 1)
        res = []
        for w in a:
            x = self.find_neigh_in_set(w, srcs)
            y = self.find_neigh_in_set(w, dsts)
            if x and y:
                res.append([x, w, y])
        return res

    def show_paths(self, srcs, dsts):
        paths = self.find_paths(srcs, dsts)
        for path in paths:
            print(path)

    def get_dis(self, dst, srcs, max_hop=3):
        vis = set()
        points = [dst]
        vis.add(dst)
        step = 0
        if dst in srcs:
            return step
        while step < max_hop:
            step += 1
            temp_points = []
            for pt in points:
                ns = self.get_neighbors(pt)
                for n in ns:
                    if n in srcs:
                        return step
                    if n in vis:
                        continue
                    vis.add(n)
                    temp_points.append(n)
            points = temp_points
        return step


def get_conceptnet(path):

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        print(len(lines))
        edges = []
        for line in lines:
            edge = line.strip().split('|||')
            edges.append(edge)

        return KnowledgeGraph(edges)


if __name__ == '__main__':

    graph = get_conceptnet()
    print(f"node num:{graph.get_node_num()}, avg deg:{graph.get_avg_deg()}")
   
    print(graph.get_hops_set(['people'], hop=1, relation=False))
    print('='*100)
    print(graph.get_hops_set(['people'], hop=1, relation=True))
    # graph.show_degs()
    # print(graph.get_hops_set(['people'], 1))
```

### Graph Vector
```
temp = embeds * mask_emb
concat_emb = torch.cat([head_embs, tail_embs], dim=1)  # [real_triple_len, hidden_size * 2]
relation_embs = self.relation_tensor(torch.tensor(relation_ids, device=device))
x = self.Wr(relation_embs)
y = torch.tanh(self.Wh(head_embs) + self.Wt(tail_embs))
betas = torch.sum(x * y, dim=-1)
```

### framework
```
get_logits()
init_weights()
```

# Retrieval-Free Knowledge-Grounded Dialogue Response Generation with Adapters

### GPT2+adapter
```
# adapter
class Adapter(nn.Module):
    def __init__(self, config, dneck):
        super().__init__()
        nx = config.n_embd if hasattr(config, "n_embd") else config.d_model
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon) if hasattr(config, "layer_norm_epsilon") else nn.LayerNorm(nx)
        self.we = nn.Linear(nx, dneck, bias=False)
        self.wd = nn.Linear(dneck, nx, bias=False)

    def forward(self, x):
        a = self.we(self.ln(x))
        m = self.wd(F.relu(a))
        output = x + m
        return output

# GPT2
hidden_states = hidden_states + feed_forward_hidden_states

if self.kadapter is not None:
    if self.num_kadapter == 1:
        hidden_states = self.kadapter(hidden_states)
    else:
        assert experts is not None
        hidden_states_list = torch.stack([self.kadapter[l](hidden_states) for l in range(self.num_kadapter)], dim=1)
        hsl_0, hsl_1, hsl_2, hsl_3 = hidden_states_list.shape
        hidden_states = torch.bmm(experts.unsqueeze(1), hidden_states_list.reshape(hsl_0, hsl_1, hsl_2*hsl_3)).reshape(hsl_0, hsl_2, hsl_3)

    if self.topic_adapter is not None:
    hidden_states = self.topic_adapter(hidden_states)
```
