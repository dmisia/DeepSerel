import ujson as json
import os
import random


file_out = os.path.join('./data/retacred', "train_tripples.json")
chunks = []

for file in os.listdir("./data/inforex_export_637/documents"):
    if file.endswith(".json"):
        # print("Convertion of " + file)
        file_in = os.path.join('./data/inforex_export_637/documents', file)

        with open(file_in, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        annotations = data.get('annotations')
        relations = data.get('relations')
        # wyciagnac zdanie
        for d in data.get('chunks')[0]:
            orth = [t['orth'] for t in d]
            for token in d:
                if len(token['relations']) >= 2:
                    token_annotation_id = token['annotations'][0]
                    token_annotation = [e for e in annotations if e['id'] == token['annotations'][0]]
                    token_annotation_type = token_annotation[0]['name']

                    if token_annotation_type == 'spatial_indicator':
                        spatial_indicator_index = token['token_id']
                        spatial_indicator_ctag = token['ctag']
                        trajectors = []
                        landmarks= []
                        [e for e in d if token['relations'] in e['relations']]

                        for rel in token['relations']:
                            # znajdź spatial objecty
                            token_objects = [e for e in d if rel in e['relations'] and e != token]
                            # znajdź relacje
                            relation_type = [e['name'] for e in relations if e['id'] == rel][0]
                            for t in token_objects:
                                if  relation_type == 'trajector':
                                    trajectors.append(t)
                                if  relation_type == 'landmark':
                                    landmarks.append(t)

                        #tworzenie trójek prawdziwych relacji
                        for tra in trajectors:
                            for lan in landmarks:
                                landmark_index = t['token_id']
                                landmark_ctag = t['ctag']
                                chunk = {
                                    "id": file,
                                    "relation": "relation",
                                    "token": orth,
                                    "ind_start": spatial_indicator_index,
                                    "ind_end": spatial_indicator_index,
                                    "ind_type": spatial_indicator_ctag,
                                    "tra_start": tra['token_id'],
                                    "tra_end": tra['token_id'],
                                    "tra_type": tra['ctag'],
                                    "lan_start": lan['token_id'],
                                    "lan_end": lan['token_id'],
                                    "lan_type": lan['ctag']
                                }
                                chunks.append(chunk)

                        # znajdź tokeny nie będące w relacji
                        non_rel_tokens = [e for e in d if e not in trajectors and e not in landmarks and e['ctag'] is not None]
                        # rzeczowniki niebędące w relacjach
                        non_rel_nouns = [e for e in non_rel_tokens if e['ctag'] is not None and e['ctag'].startswith('subst')]
                        # najbliższy indeks w prawo i w lewo od indicatora
                        left_non_rel_tokens = [e for e in non_rel_tokens if e['token_id'] < spatial_indicator_index]
                        right_non_rel_tokens = [e for e in non_rel_tokens if e['token_id'] > spatial_indicator_index]

                        # jeden przykład z tym samym SP IND, lecz innymi rzeczownikami
                        if len(non_rel_nouns) > 1:
                            no_rel_tra = random.choice(non_rel_nouns)
                            no_rel_lan = random.choice(non_rel_nouns)
                        else:
                            no_rel_tra = random.choice(non_rel_tokens)
                            no_rel_lan = random.choice(non_rel_tokens)
                        chunk = {
                            "id": file,
                            "relation": "no_relation",
                            "token": orth,
                            "ind_start": spatial_indicator_index,
                            "ind_end": spatial_indicator_index,
                            "ind_type": spatial_indicator_ctag,
                            "tra_start": no_rel_tra['token_id'],
                            "tra_end": no_rel_tra['token_id'],
                            "tra_type": no_rel_tra['ctag'],
                            "lan_start": no_rel_lan['token_id'],
                            "lan_end": no_rel_lan['token_id'],
                            "lan_type": no_rel_lan['ctag']
                        }
                        chunks.append(chunk)

                        # jeden przykład z tym samym SP IND, i innymi wyrazami ale blisko z prawej i lewej
                        if len(right_non_rel_tokens) != 0:
                            if len(right_non_rel_tokens) > 4:
                                no_rel_lan = random.choice(right_non_rel_tokens[:4])
                            else:
                                no_rel_lan = random.choice(right_non_rel_tokens)
                        else:
                            no_rel_lan = random.choice(non_rel_tokens)
                        if len(left_non_rel_tokens) != 0:
                            if len(left_non_rel_tokens) > 4:
                                no_rel_tra = random.choice(left_non_rel_tokens[-4:])
                            else:
                                no_rel_tra = random.choice(left_non_rel_tokens)
                        else:
                            no_rel_tra = random.choice(non_rel_tokens)

                        chunk = {
                            "id": file,
                            "relation": "no_relation",
                            "token": orth,
                            "ind_start": spatial_indicator_index,
                            "ind_end": spatial_indicator_index,
                            "ind_type": spatial_indicator_ctag,
                            "tra_start": no_rel_tra['token_id'],
                            "tra_end": no_rel_tra['token_id'],
                            "tra_type": no_rel_tra['ctag'],
                            "lan_start": no_rel_lan['token_id'],
                            "lan_end": no_rel_lan['token_id'],
                            "lan_type": no_rel_lan['ctag']
                        }
                        chunks.append(chunk)

                        # jeden przykład z randomowymi danymi
                        no_rel_tra = random.choice(non_rel_tokens)
                        no_rel_lan = random.choice(non_rel_tokens)
                        no_rel_ind = random.choice(non_rel_tokens)
                        chunk = {
                            "id": file,
                            "relation": "no_relation",
                            "token": orth,
                            "ind_start": no_rel_ind['token_id'],
                            "ind_end": no_rel_ind['token_id'],
                            "ind_type": no_rel_ind['ctag'],
                            "tra_start": no_rel_tra['token_id'],
                            "tra_end": no_rel_tra['token_id'],
                            "tra_type": no_rel_tra['ctag'],
                            "lan_start": no_rel_lan['token_id'],
                            "lan_end": no_rel_lan['token_id'],
                            "lan_type": no_rel_lan['ctag']
                        }
                        chunks.append(chunk)

with open(file_out, "w", encoding="utf-8") as outfile:
    json.dump(chunks, outfile, indent=4, sort_keys=False, ensure_ascii=False)
