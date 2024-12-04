import json


class Translator:
    def __init__(self, languages, default_lang='ru'):
        self.languages = languages
        self.translations = {}
        self.current_lang = default_lang
        self.load_translations()

    def load_translations(self):
        for lang in self.languages:
            lang_code = lang['code']
            try:
                with open(f'localizations/lang_{lang_code}.json', 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            except FileNotFoundError:
                print(f"Translation file for language '{lang_code}' not found.")
                self.translations[lang_code] = {}

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_lang = lang_code
        else:
            print(f"Language '{lang_code}' not supported.")

    def get(self, key, **kwargs):
        return self.translations.get(self.current_lang, {}).get(key, key).format(**kwargs)
