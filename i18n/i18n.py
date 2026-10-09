import json
import locale
import os
from tools.file_io import read_text


def load_language_list(language):
    return json.loads(read_text(f"./i18n/locale/{language}.json"))


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            # getdefaultlocale() is deprecated and scheduled for removal.
            # Honor the same environment preference without changing process locale.
            language = None
            for variable in ("LC_ALL", "LC_CTYPE", "LANG", "LANGUAGE"):
                value = os.environ.get(variable, "").split(":")[0]
                if value:
                    language = value.split(".")[0].split("@")[0]
                    break
            if not language:
                language = locale.getlocale()[0]
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
