from tqdm import tqdm
import ujson as json


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

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        input_format = self.args.input_format
        if input_format == 'entity_mask':
            subj_type = '[SUBJ-{}]'.format(subj_type)
            obj_type = '[OBJ-{}]'.format(obj_type)
            for token in (subj_type, obj_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker_punct':
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == os:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == 'entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_os + 1


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)



        self.LABEL_TO_ID = {'no_relation': 0,
                       'affiliation': 1,
                       'alias': 2,
                       'composition': 3,
                       'creator': 4,
                       'location': 5,
                       'nationality': 6,
                       'neighbourhood': 7,
                       'origin': 8}

    def read(self, file_in):
        features = []
        with open(file_in, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features


class RETACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

        self.LABEL_TO_ID = {'no_relation': 0,
                            'affiliation': 1,
                            'alias': 2,
                            'composition': 3,
                            'creator': 4,
                            'location': 5,
                            'nationality': 6,
                            'neighbourhood': 7,
                            'origin': 8}

    def read(self, file_in):
        features = []
        with open(file_in, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features
