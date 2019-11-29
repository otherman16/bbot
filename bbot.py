import slack
import os
import logging
import ssl as ssl_lib
import certifi
import random
from nlp_model import NLPModel

punctuation_marks = [".", ",", "!", "?"]


def drop_punctuation(text):
    for p in punctuation_marks:
        text = text.replace(p, "")
    return text


class BokshNet(object):

    def __init__(self, dataset_path="boksh.txt"):
        self.model = NLPModel()
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line:
                    line = line.rstrip()
                    dataset.append(line)

        self.model.fit(dataset)

    def predict(self, query):
        query = drop_punctuation(query)
        if query == "":
            return "Мне нечего тут добавить :boksh:"

        score, texts = self.model.predict(query)
        if score == 0:
            return "Мне нечего тут добавить :boksh:"
        else:
            return texts[random.randint(0, len(texts))]


class FreqModerator:

    def __init__(self):
        self.freq = 3

    def ready2answer(self):
        return random.randrange(self.freq) == self.freq - 1

    def set_freq(self, f):
        self.freq = f
        pass


boksh_net = BokshNet("boksh.txt")
fmod = FreqModerator()


# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@slack.RTMClient.run_on(event="message")
def message(**payload):
    """Display the onboarding welcome message after receiving a message
    that contains "start".
    """
    data = payload["data"]
    web_client = payload["web_client"]
    channel_id = data.get("channel")
    user_id = data.get("user")
    text = data.get("text")
    thread_ts = data.get("ts")

    if user_id is not None:
        if user_id != "USLACKBOT":
            if "bokshbot" in text.lower():
                if "freq" in text.lower():
                    _freq = None
                    try:
                        last_word = text.split()[-1]
                        _freq = int(last_word)
                    except Exception as e:
                        e.args += (last_word)
                    fmod.set_freq(_freq)
                    # if _freq is not None:
                        # response_freq = _freq
                web_client.chat_postMessage(
                    channel=channel_id,
                    text=boksh_net.predict(text)
                )
            elif fmod.ready2answer():
                web_client.chat_postMessage(
                    channel=channel_id,
                    text=boksh_net.predict(text)
                )
    pass


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    rtm_client = slack.RTMClient(token=slack_token, ssl=ssl_context)
    rtm_client.start()
    pass


TEXT = '''
Отличие в том, что стеммер (конкретная реализация алгоритма стемминга – прим.переводчика) действует без знания 
контекста и, соответственно, не понимает разницу между словами, которые имеют разный смысл в зависимости от части речи. 
Однако у стеммеров есть и свои преимущества: их проще внедрить и они работают быстрее. 
Плюс, более низкая «аккуратность» может не иметь значения в некоторых случаях.
'''


def test():
    # text = open('./anna.txt', 'r', encoding='utf-8').read().lower()
    # text = TEXT.lower()
    # model = NLPModel()
    # words = model.word_tokenize(text, preserve_line=False)
    # clean_words = model.remove_stopwords(words)
    # print(clean_words)
    # print()
    # stem_words = model.stem_words(clean_words)
    # print(stem_words)
    # print()
    # bow = model.create_bow(stem_words)
    # print(bow)
    # dataset = []
    # with open("boksh.txt", 'r') as f:
    #     for line in f:
    #         if line:
    #             line = line.rstrip()
    #             dataset.append(line)
    #
    # model.fit(dataset)
    # score, text = model.predict("зарплата")
    print(boksh_net.predict("утка"))


if __name__ == "__main__":
    # main()
    test()
