import slack
import os
import logging
import ssl as ssl_lib
import certifi
import random

punctuation_marks = [".", ",", "!", "?"]


def drop_punctuation(text):
    for p in punctuation_marks:
        text = text.replace(p, "")
    return text


class BokshNet(object):

    def __init__(self, dataset_path="boksh.txt"):
        self.dataset = []
        with open(dataset_path, 'r') as f:
            for line in f:
                src_line = line[:]
                if line:
                    line = line.rstrip().lower()
                    self.dataset.append({'text': src_line, 'bow': self.get_bow(line)})

    @staticmethod
    def get_bow(text):
        text = drop_punctuation(text)
        bow = dict()
        for word in text.split():
            if word not in bow:
                bow[word] = 0
            bow[word] += 1
        return bow

    def predict(self, query):
        query = drop_punctuation(query)
        if query == "":
            return "Мне нечего тут добавить :boksh:"
        rank = []
        for data in self.dataset:
            cur_rank = 0
            for w in query.lower().split(" "):
                if w in data["bow"].keys():
                    cur_rank += data["bow"][w]
            rank.append(cur_rank)
        ids = [x for x in range(len(self.dataset))]
        id_rank = zip(ids, rank)
        sorted_id_rank = sorted(id_rank, key=lambda x: x[1], reverse=True)
        sorted_id_rank = list(zip(*sorted_id_rank))
        sorted_ids, sorted_ranks = sorted_id_rank
        top = []
        top.append(sorted_ids[0])
        next = 1
        while sorted_ranks[next] == sorted_ranks[0]:
            top.append(sorted_ids[next])
            if next == len(sorted_ids) - 1:
                break
            else:
                next += 1

        out_id = top[random.randrange(len(top))]
        return self.dataset[out_id]["text"]


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


if __name__ == "__main__":
    main()
