"""바이트 수준 BPE 토크나이저 학습 (과제 PDF 2.4~2.5, 9쪽 Problem train_bpe)

뼈대만 있는 파일입니다. TODO를 위에서부터 직접 채우세요.
"""
from collections import Counter, defaultdict
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_word(word, pair):
    new = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1])==pair:
            new.append(word[i] + word[i+1])
            i += 2
        else:
            new.append(word[i])
            i += 1
    return tuple(new)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # TODO 1. 파일 읽기
    #   input_path의 텍스트를 UTF-8 문자열로 읽는다
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # TODO 2. 어휘 초기화
    #   vocab: dict[int, bytes]
    #   특수 토큰(UTF-8 bytes로)과 바이트 256개에 ID를 매긴다
    vocab = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # TODO 3. 특수 토큰으로 문서 나누기 (8쪽)
    #   re.escape로 감싼 특수 토큰들을 "|"로 이어 re.split
    #   특수 토큰 목록이 비어 있을 때는 나누지 않고 통째로 쓴다
    if special_tokens:
        split_pat = "|".join(re.escape(tok) for tok in special_tokens)
        docs = re.split(split_pat, text)
    else:
        docs = [text]

    # TODO 4. 사전 토큰화하고 빈도 세기 (6쪽)
    #   나눈 조각마다 re.finditer(PAT, 조각)
    #   각 덩어리를 (b'l', b'o', b'w') 같은 bytes 튜플로 바꿔 빈도를 센다
    word_counts = Counter()
    for doc in docs:
        for m in re.finditer(PAT, doc):
            word = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            word_counts[word]+=1
    

    # TODO 5. 병합 반복 (7쪽 예제와 같은 방식)
    #   모든 쌍 세기
    pair_counts = Counter()
    pair_words = defaultdict(set)
    for word, c in word_counts.items():
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += c
            pair_words[(a, b)].add(word)

    #   가장 큰 쌍 골라 기록
    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        for word in list(pair_words[best]):
            if word not in word_counts:
                continue
            c = word_counts.pop(word)
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] -= c
                if pair_counts[(a, b)] == 0:
                    del pair_counts[(a, b)]
            new = merge_word(word, best)
            word_counts[new] = word_counts.get(new, 0) + c
            for a, b in zip(new, new[1:]):
                pair_counts[(a, b)] += c
                pair_words[(a, b)].add(new)
        del pair_words[best]

    # TODO 6. vocab, merges 반환
    return vocab, merges
