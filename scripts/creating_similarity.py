import yaml 
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from collections import defaultdict
from nltk.corpus import wordnet as wn
from data.imagenet_classnames import name_map, folder_label_map


synset_human = folder_label_map
index_human = name_map
human_synset = { v: k for k, v in synset_human.items() }
index_synset = { k: human_synset[v] for k, v in index_human.items()}


# with open("data/imagenette2/synset_human.txt", "r") as f:    
#     synset_human = f.read().splitlines()
#     synset_human = dict(line.split(maxsplit=1) for line in synset_human)

# with open('data/synset_closest_idx.yaml', 'r') as file:
#     synset_closest_idx = yaml.safe_load(file)

# with open('data/index_synset.yaml', 'r') as file:
#     index_synset = yaml.safe_load(file)

# Accessing the data
print(index_synset)
synset_index = {v: k for k, v in index_synset.items()}

def get_sim_synset(synset_id_1, synset_id_2):
    # Map the synset IDs to WordNet synsets
    synset_1 = wn.synset_from_pos_and_offset(synset_id_1[0], int(synset_id_1[1:]))
    synset_2 = wn.synset_from_pos_and_offset(synset_id_2[0], int(synset_id_2[1:]))

    # Calculate the Wu-Palmer Similarity score
    similarity_score = synset_1.lch_similarity(synset_2)
    return similarity_score

#get the closest two words for each synset
synset_closest_syn = defaultdict(list)
synset_closest_idx = defaultdict(list)
synet_closest_hum = defaultdict(list)

for k, v in index_synset.items():
    print(k, v)
    similarity_scores = []
    for k2, v2 in index_synset.items():
        #print(k, v, k2, v2)
        if k!=k2:
            # if synset_human[v] == 'kite':
            #     print('getting sim for:', synset_human[v2])
            similarity_scores.append((v2, get_sim_synset(v, v2)))

            
    # sort the similarity scores in ascending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # if synset_human[v] == 'kite':
    #     print(synset_human[v2])
    #     lista = [ (synset_human[x[0]], x[1]) for x in similarity_scores[:50] ]
    #     print(lista)
    #     exit()
    # add the closest five keys to the list for this synset
    closest_keys = [x[0] for x in similarity_scores[:5]]
    print("closest keys", closest_keys, '\n')
    synset_closest_syn[v] = closest_keys
    synset_closest_idx[k] = [synset_index[x] for x in closest_keys]
    synet_closest_hum[synset_human[v]] = [synset_human[x] for x in closest_keys]
    print(k, [synset_index[x] for x in closest_keys], '\n')
    print(synset_human[v], synet_closest_hum[synset_human[v]], '\n')
    #print(synset_human[v], [synset_human[x] for x in closest_keys])

    #break 

#save synset_closest_syn and synset_closest_idx to yaml file
with open('data/synset_closest_syn.yaml', 'w') as file:
    documents = yaml.dump(dict(synset_closest_syn), file)

with open('data/synset_closest_idx.yaml', 'w') as file:
    documents = yaml.dump(dict(synset_closest_idx), file)

with open('data/synset_closest_hum.yaml', 'w') as file:
    documents = yaml.dump(dict(synet_closest_hum), file)