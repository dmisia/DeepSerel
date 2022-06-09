import ujson as json
import os

file_out = os.path.join('./data', "processed_inforex_export_637.json")
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
                    spatial_indicator_index = token['token_id']
                    spatial_indicator_ctag = token['ctag']
                    token_annotation_id = token['annotations'][0]
                    token_annotation = [e for e in annotations if e['id'] == token['annotations'][0]]
                    token_annotation_type = token_annotation[0]['name']

                    if token_annotation_type == 'spatial_indicator':

                        for rel in token['relations']:
                            # znajdź spatial objecty
                            token_objects = [e for e in d if rel in e['relations'] and e != token]
                            # znajdź relacje
                            relation_type = [e['name'] for e in relations if e['id'] == rel][0]
                            for t in token_objects:
                                obj_ann_type = [e for e in annotations if e['id'] == t['annotations'][0]][0]['name']
                                if obj_ann_type == 'spatial_object' and relation_type == 'trajector':
                                    trajector_index = t['token_id']
                                    trajector_ctag = t['ctag']
                                    chunk = {
                                        "id": file,
                                        "relation": "trajector_indicator",
                                        "token" : orth,
                                        "ind_start": spatial_indicator_index,
                                        "ind_end": spatial_indicator_index,
                                        "obj_start": trajector_index,
                                        "obj_end": trajector_index,
                                        "ind_type": spatial_indicator_ctag,
                                        "obj_type": trajector_ctag
                                    }
                                    chunks.append(chunk)
                                if obj_ann_type == 'spatial_object' and relation_type == 'landmark':
                                    landmark_index = t['token_id']
                                    landmark_ctag = t['ctag']
                                    chunk = {
                                        "id": file,
                                        "relation": "landmark_indicator",
                                        "token" : orth,
                                        "ind_start": spatial_indicator_index,
                                        "ind_end": spatial_indicator_index,
                                        "obj_start": landmark_index,
                                        "obj_end": landmark_index,
                                        "ind_type": spatial_indicator_ctag,
                                        "obj_type": landmark_ctag
                                    }
                                    chunks.append(chunk)
                                #dodawanie no_relation


with open(file_out, "w", encoding="utf-8") as outfile:
    json.dump(chunks, outfile, indent=4, sort_keys=False, ensure_ascii=False)