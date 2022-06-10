from tqdm import tqdm
import ujson as json
import itertools
import operator

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = []
        if self.args.input_format == 'entity_marker':
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        self.tokenizer.add_tokens(self.new_tokens)
        if self.args.input_format not in ('entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct'):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, ind_type, tra_type, lan_type, i_s, i_e, t_s, t_e, l_s, l_e):
        """
        Implement the following input formats:
            - entity_mask: [IND-NER], [OBJ-NER].
            - entity_marker: [E1] indect [/E1], [E2] object [/E2].
            - entity_marker_punct: @ indect @, # object #.
            - typed_entity_marker: [IND-NER] indect [/IND-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * indect ner type * indect @, # ^ object ner type ^ object #
        """
        sents = []
        input_format = self.args.input_format
        if input_format == 'entity_mask':
            ind_type = '[IND-{}]'.format(ind_type)
            tra_type = '[OBJ-{}]'.format(tra_type)
            for token in (ind_type, tra_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker':
            ind_start = '[IND-{}]'.format(ind_type)
            ind_end = '[/IND-{}]'.format(ind_type)
            obj_start = '[OBJ-{}]'.format(tra_type)
            obj_end = '[/OBJ-{}]'.format(tra_type)
            for token in (ind_start, ind_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker_punct':
            ind_type = self.tokenizer.tokenize(ind_type.replace("_", " ").lower())
            tra_type = self.tokenizer.tokenize(tra_type.replace("_", " ").lower())
            lan_type = self.tokenizer.tokenize(lan_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if i_s <= i_t <= i_e or t_s <= i_t <= t_e:
                    tokens_wordpiece = []
                    if i_t == i_s:
                        new_i_s = len(sents)
                        tokens_wordpiece = [ind_type]
                    if i_t == t_s:
                        new_t_s = len(sents)
                        tokens_wordpiece = [tra_type]

            elif input_format == 'entity_marker':
                if i_t == i_s:
                    new_i_s = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == i_e:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == t_s:
                    new_t_s = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == t_e:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == i_s:
                    new_i_s = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == i_e:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == t_s:
                    new_t_s = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == t_e:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker':
                if i_t == i_s:
                    new_i_s = len(sents)
                    tokens_wordpiece = [ind_start] + tokens_wordpiece
                if i_t == i_e:
                    tokens_wordpiece = tokens_wordpiece + [ind_end]
                if i_t == t_s:
                    new_t_s = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == t_e:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct':
                if i_t == i_s:
                    new_i_s = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + ind_type + ['*'] + tokens_wordpiece
                if i_t == i_e:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == t_s:
                    new_t_s = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + tra_type + ['^'] + tokens_wordpiece
                if i_t == t_e:
                    tokens_wordpiece = tokens_wordpiece + ["#"]
                if i_t == l_s:
                    new_l_s = len(sents)
                    tokens_wordpiece = ["%"] + [':'] + lan_type + [':'] + tokens_wordpiece
                if i_t == l_e:
                    tokens_wordpiece = tokens_wordpiece + ["%"]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_i_s + 1, new_t_s + 1, new_l_s + 1


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)



        self.LABEL_TO_ID = {'no_relation': 0,
                            'relation': 1}

    def read(self, file_in):
        features = []
        with open(file_in, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['ind_start'], d['ind_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_i_s, new_t_s, new_l_s= self.tokenize(tokens, d['ind_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_i_s,
                'os': new_t_s,
            }

            features.append(feature)
        return features


class RETACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

        self.LABEL_TO_ID = {'no_relation': 0,
                            'relation': 1}

    def read(self, file_in):
        features = []
        with open(file_in, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # for d in tqdm(data.get('chunks')[0]):
        for d in tqdm(data):
            i_s, i_e = d['ind_start'], d['ind_end']
            t_s, t_e = d['tra_start'], d['tra_end']
            l_s, l_e = d['lan_start'], d['lan_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_i_s, new_t_s, new_l_s = self.tokenize(tokens, d['ind_type'], d['tra_type'], d['lan_type'], i_s, i_e, t_s, t_e, l_s, l_e)
            rel = self.LABEL_TO_ID[d['relation']]

            sample_id = d['id']

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'i_s': new_i_s,
                't_s': new_t_s,
                'l_s': new_l_s,
                'sample_ids': sample_id,
            }

            features.append(feature)
        return features

    def readTest(self, file_in):
            features = []
            with open(file_in, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # generate
            get_item = operator.itemgetter('token')
            grouped_data = [list(g) for k, g in itertools.groupby(sorted(data, key=get_item), get_item)]

            for group in grouped_data:
                true_triples = []
                for sentence in group:
                    triple = []
                    triple.append(sentence['ind_start'])
                    triple.append(sentence['tra_start'])
                    triple.append(sentence['lan_start'])
                    true_triples.append(triple)
                # check number of tokens (words)
                sentence = group[0]
                num_indexes = len(sentence['token'])-1
                triple_combinations = [list(x) for x in itertools.combinations(list(range(0,num_indexes)), 3)]
                for t in triple_combinations:
                    if t not in true_triples:
                        chunk = {
                            "id": sentence['id'],
                            "relation": "no_relation",
                            "token": sentence['token'],
                            "ind_start": t[0],
                            "ind_end": t[0],
                            "ind_type": sentence['ctags'][t[0]],
                            "tra_start": t[1],
                            "tra_end": t[1],
                            "tra_type": sentence['ctags'][t[1]],
                            "lan_start": t[2],
                            "lan_end": t[2],
                            "lan_type": sentence['ctags'][t[2]]
                        }
                        # group.append(chunk)
                        data.append(chunk)

            for d in tqdm(data):
                i_s, i_e = d['ind_start'], d['ind_end']
                t_s, t_e = d['tra_start'], d['tra_end']
                l_s, l_e = d['lan_start'], d['lan_end']

                tokens = d['token']
                tokens = [convert_token(token) for token in tokens]

                input_ids, new_i_s, new_t_s, new_l_s = self.tokenize(tokens, d['ind_type'], d['tra_type'], d['lan_type'], i_s, i_e, t_s, t_e, l_s, l_e)
                rel = self.LABEL_TO_ID[d['relation']]

                sample_id = d['id']

                feature = {
                    'input_ids': input_ids,
                    'labels': rel,
                    'i_s': new_i_s,
                    't_s': new_t_s,
                    'l_s': new_l_s,
                    'sample_ids': sample_id,
                }

                features.append(feature)
            return features
