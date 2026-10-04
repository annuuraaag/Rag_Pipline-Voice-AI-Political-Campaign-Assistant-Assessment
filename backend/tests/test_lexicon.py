import pytest

from app.lexicon import detect_topic, topic_scores


@pytest.mark.parametrize(
    "prose",
    [
        "It is what it is, and it will be decided later.",
        "Contact the office if you want to register for the newsletter.",
        "The security of the system matters; social media power is overrated.",
        "Keep it cold. Storage space is limited.",
    ],
)
def test_ordinary_prose_gets_no_topic(prose):
    assert topic_scores(prose) == {}
    assert detect_topic(prose) is None


def test_it_pronoun_is_not_employment():
    text = "It is planned that it will open soon, and it has support. Hospitals and doctors will benefit."
    assert detect_topic(text) == "healthcare"


def test_multi_word_phrases_match():
    assert topic_scores("A chilli cold storage complex")["agriculture"] == 2
    assert topic_scores("An IT and electronics corridor in the IT sector")["employment"] == 2
    assert topic_scores("Old-age pensions and social security")["welfare"] == 2
    assert topic_scores("Check voter registration at the campaign office")["campaign_info"] == 2
    assert topic_scores("Phrases may span\nline   breaks: cold\nstorage")["agriculture"] == 1


def test_crop_insurance_is_agriculture_not_healthcare():
    scores = topic_scores("Weather-linked crop insurance with claims settled in 45 days")
    assert scores == {"agriculture": 1}
    assert topic_scores("cashless health insurance")["healthcare"] == 1  # one phrase hit, not two


def test_keywords_match_whole_words_only():
    assert topic_scores("hospitality and coldness") == {}
    assert topic_scores("self-help groups")["welfare"] == 1
