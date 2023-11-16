import re


def word_count(text: str, return_word_list=False) -> int | tuple[int, list[str]]:
    "Count words, punctuation counts as a word."
    pattern = re.compile(r"\w+|[^\w\s]")
    word_list = pattern.findall(text)
    if not return_word_list:
        return len(word_list)
    else:
        return len(word_list), word_list

def punctuation_convert(text, direction='ch-en'):
    if direction=='ch-en':
        table = {ord(f):ord(t) for f,t in zip(
            u'！？。，；：“”‘’【】「」〈〉〜（）％＃＠＆１２３４５６７８９０',
            u'!?.,;:\"\"\'\'[]{}<>~()%#@&1234567890')}
    else:
        table = {ord(f):ord(t) for f,t in zip(
            u'!?.,;:\"\"\'\'[]<>~()',
            u'！？。，；：“”‘’【】〈〉～（）'
            )}
    return text.translate(table)