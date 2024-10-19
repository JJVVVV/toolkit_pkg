def lines_to_ngrams(lines, n=3):
    ngrams = []
    for s in lines:
        words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
        ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    return ngrams

