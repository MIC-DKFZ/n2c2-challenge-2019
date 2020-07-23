import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/klaus/repositories/k656s/googleapplicationcredentials/N2C2-2dbc3c91e74b.json"

def run_quickstart():
    # [START translate_quickstart]
    # Imports the Google Cloud client library
    from google.cloud import translate

    # Instantiates a client
    translate_client = translate.Client()

    # The text to translate
    text = u'Hello, world!'
    # The target language
    target = 'ru'

    # Translates some text into Russian
    translation = translate_client.translate(
        text,
        target_language=target)

    print(u'Text: {}'.format(text))
    print(u'Translation: {}'.format(translation['translatedText']))
    # [END translate_quickstart]


if __name__ == '__main__':
    run_quickstart()