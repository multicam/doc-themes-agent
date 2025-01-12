import json
from models.theme import Theme

class ThemeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Theme):
            return obj.to_dict()
        return super().default(obj)