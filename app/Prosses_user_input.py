import spacy
import csv
import emoji



class ProssesUserInput:

    def __init__(self, abervations_file: str = "app/app_data/Abbreviations and Slang.csv"):
        self.nlp = spacy.load("en_core_web_sm")
        self.abervations_dict = self._load_dict_from_csv(abervations_file)

    def _set_nlp(self, raw_user_input: str):
        self.doc = self.nlp(raw_user_input)

    def _tokenize(self, raw_user_input: str):
        """
        Tokenizes the user input while considering a character limit.
        """
        self._set_nlp(raw_user_input)

        if self._get_token_processing_limit(raw_user_input) == -1:
            return []
        tokens = [token.text for token in self.doc]

        return tokens

    def _get_token_processing_limit(self, str_user_input: str):
        """
        Determines a processing limit for long inputs.
        """
        char_amount = len(str_user_input)

        if char_amount > 650:
            return -1

    def _standardize_user_input(self, tokenized_user_input: list):
        """
        Standardizes user input by handling abbreviations, emojis, and case normalization.
        """
        if isinstance(tokenized_user_input, str):
            tokenized_user_input = self._tokenize(tokenized_user_input, dynamic_tokenization=False)

        standardized_tokens = []

        for token in tokenized_user_input:
            if emoji.is_emoji(token):
                standardized_tokens.append(emoji.demojize(token))
            else:
                token = token.lower()
                temp_token = self._standardize_abbreviations(token)
                if temp_token is not None  and len(temp_token) == 1:
                    standardized_tokens.append(temp_token)
                elif temp_token is not None and len(temp_token) > 1:
                    standardized_tokens.extend(temp_token.split())
            
        return standardized_tokens

    def _standardize_abbreviations(self, token: str):
        return self.abervations_dict.get(token, token)

    def _load_dict_from_csv(self, input_csv: str) -> dict:
        """
        Loads a dictionary of abbreviations and slang terms from a CSV file.
        """
        output_dict = {}
        with open(input_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            if len(fieldnames) < 2:
                raise ValueError("CSV must have at least two columns for key-value mapping.")

            for row in reader:
                output_dict[row[fieldnames[0]]] = row[fieldnames[1]]
        return output_dict

    def _process_entities(self, user_input: str, standardized_tokens: list):
        """
        Extracts entities from user input and maps them to standardized tokens.
        """
        useful_entity_labels = [
        "ORG",          # Companies, agencies, institutions
        "PERSON",       # People, including fictional
        "GPE",          # Geopolitical Entities (countries, cities, states)
        "PRODUCT",      # Objects, vehicles, foods, etc.
        "EVENT",        # Named hurricanes, battles, wars, sports events
        "LOC",          # Non-GPE locations, mountain ranges, bodies of water
        "WORK_OF_ART",  # Titles of books, songs, etc.
        "LANGUAGE",     # Any named language (e.g., English, Spanish)
        "NORP",         # Nationalities, religious, or political groups
        "DATE"          # Absolute or relative dates or periods (for experimentation)
     ]
        self._set_nlp(user_input)
        
        entities_defs = []
        
        for word in self.doc.ents:
            # Check if the entity label is in the list of useful labels
            if word.label_ in useful_entity_labels:
                
                # Standardize the entity
                standardized_entity_list = self._standardize_user_input([word.text])
                if standardized_entity_list:
                    standardized_entity = standardized_entity_list[0]
                else:
                    standardized_entity = ""
                ent_index = self._get_entity_index(standardized_entity, standardized_tokens)

                if ent_index is not None:
                    temp_list = [word.text, spacy.explain(word.label_), word.label_]
                    temp_list.append(ent_index)
                    entities_defs.append(temp_list)

        return entities_defs

    def _get_entity_index(self, standardized_entity: str, standardized_tokens: list):
        """
        Finds the index of an entity within the standardized token list.
        """
        for i in standardized_tokens:
            if i == standardized_entity:
                return standardized_tokens.index(i)
        return None

    def _bert_segment(self, standardized_input: list, entities_and_indexes: list):
        """
        Segments the standardized tokens for BERT input.
        """
        
        bert_input = ""
        for token in standardized_input:
            if token in [".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "'", '"', "“", "”", "‘", "’","'","'",]:
                bert_input += token
            else:
                bert_input += " " + token
                
        bert_input += " <ENT>"

        for i in range(len(entities_and_indexes)):
            bert_input += " entity: " + entities_and_indexes[i][0] + ", type: " + entities_and_indexes[i][2]

            if i < len(entities_and_indexes) - 1:
                bert_input += ";"
                
        bert_input += " </ENT>"

        return bert_input

    def process_user_input(self, raw_user_input: str):
        """
        Processes the user query by tokenizing, standardizing, and extracting entities.
        """
        # Tokenization
        tokenized_user_input = self._tokenize(raw_user_input)
        
        if not tokenized_user_input:
            return {
                "tokenized": [],
                "standardized": "",
                "entities": [],
                "BERT_Input": ""
            }
        # Standardization
        standardized_input = self._standardize_user_input(tokenized_user_input)
        
        # Entity Extraction
        entities_and_indexes = self._process_entities(raw_user_input, standardized_input)

        # BERT segmentation
        bert_input = self._bert_segment(standardized_input, entities_and_indexes)
        
        return {
            "tokenized": tokenized_user_input,
            "standardized": " ".join(standardized_input),
            "entities": entities_and_indexes,
            "BERT_Input": bert_input
        }