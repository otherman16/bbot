import slack
import os
import logging
import ssl as ssl_lib
import certifi
import random
from nlp_model import NLPModel, SiteModel


class BokshBot(object):

    def __init__(self, dataset_path="boksh.txt"):
        self.model = NLPModel()
        self.site = SiteModel()
        dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line:
                    line = line.rstrip()
                    dataset.append(line)

        self.model.fit(dataset)

    def answer(self, query):
        if query == "":
            return "Мне нечего тут добавить :boksh:"

        score, texts = self.model.predict(query)
        if score == 0:
            return [f"Я знаю сайт {self.site(query)}! Поищите там! :boksh:",
                    "Мне нечего тут добавить :boksh:"][random.randrange(2)]
        else:
            return texts[random.randrange(len(texts))]


class FreqModerator:

    def __init__(self):
        self.__freq = 3

    def ready2answer(self):
        return random.randrange(self.__freq) == self.__freq - 1

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, v):
        if v > 0:
            self.__freq = v


bbot = BokshBot("boksh.txt")
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
    rtm_client = payload["rtm_client"]
    channel_id = data["channel"]
    user_id = data.get("user", None)
    text = data.get("text", '')
    thread_ts = data["ts"]

    if user_id is not None:
        if user_id != "USLACKBOT":
            if "bokshbot" in text.lower():
                if "freq" in text.lower():
                    last_word = text.split()[-1]
                    try:
                        _freq = int(last_word)
                        fmod.freq = _freq
                        web_client.chat_postMessage(
                            channel=channel_id,
                            text=":boksh::+1:"
                        )
                    except ValueError:
                        web_client.chat_postMessage(
                            channel=channel_id,
                            text=":boksh::-1:"
                        )
                else:
                    web_client.chat_postMessage(
                        channel=channel_id,
                        text=bbot.answer(text)
                    )
            elif fmod.ready2answer():
                web_client.chat_postMessage(
                    channel=channel_id,
                    text=bbot.answer(text)
                )


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    ssl_context = ssl_lib.create_default_context(cafile=certifi.where())
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    freq = os.environ["SLACK_BOT_FREQ"]
    fmod.freq = int(freq)
    rtm_client = slack.RTMClient(token=slack_token, ssl=ssl_context)
    rtm_client.start()


if __name__ == "__main__":
    main()
