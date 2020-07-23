from translate import Translator
translator = Translator(to_lang="de")
translation = translator.translate("This is a house.")
print(translation)