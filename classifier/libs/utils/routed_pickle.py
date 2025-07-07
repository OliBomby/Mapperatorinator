import pickle
from typing import Dict


class Unpickler(pickle.Unpickler):
    load_module_mapping: Dict[str, str] = {
        'osuT5.tokenizer.event': 'osuT5.osuT5.event',
        'libs.tokenizer.event': 'classifier.libs.tokenizer.event',
        'libs.tokenizer.tokenizer': 'classifier.libs.tokenizer.tokenizer',
        'osuT5.event': 'osuT5.osuT5.event',
        'libs.event': 'classifier.libs.tokenizer.event',
        'libs.tokenizer': 'classifier.libs.tokenizer.tokenizer',
    }

    def find_class(self, mod_name, name):
        mod_name = self.load_module_mapping.get(mod_name, mod_name)
        return super().find_class(mod_name, name)
