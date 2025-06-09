import pickle
from typing import Dict


class Unpickler(pickle.Unpickler):
    load_module_mapping: Dict[str, str] = {
        'osuT5.tokenizer.event': 'osuT5.osuT5.tokenizer.event'
    }

    def find_class(self, mod_name, name):
        mod_name = self.load_module_mapping.get(mod_name, mod_name)
        return super().find_class(mod_name, name)
