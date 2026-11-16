"""2.6 Problem (tokenizer): BPE 토크나이저 (encode / decode).

학습 결과(vocab, merges)를 받아 텍스트 <-> 토큰 ID 변환을 한다.
위치: cs336_basics/tokenizer.py
"""
from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator

import regex as re

# 학습 때와 같은 GPT-2 사전 토큰화 정규식
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 특수 토큰 없이 버퍼가 이보다 커지면 강제로 비운다 (글자 수 기준, 약 1MB)
MAX_BUFFER = 1 << 20


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)  # 원본을 건드리지 않도록 복사
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 특수 토큰이 vocab에 없으면 끝에 새 ID로 추가
        existing = set(self.vocab.values())
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)

        # 바이트열 -> ID (vocab을 뒤집은 사전)
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        # 쌍 -> 병합 순위 (작을수록 먼저 만들어진 병합)
        self.rank = {pair: i for i, pair in enumerate(merges)}

        # 특수 토큰으로 자르는 정규식. 긴 것부터 넣어야 겹치는 토큰에서 긴 쪽이 이긴다
        # 괄호로 감싸면 re.split 결과에 잘라낸 특수 토큰도 남는다
        if self.special_tokens:
            toks = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = "(" + "|".join(re.escape(t) for t in toks) + ")"
        else:
            self.special_pat = None

        # 같은 사전 토큰은 결과가 항상 같으니 한 번 계산하면 저장해 둔다
        self.cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # train_bpe_run.py가 pickle로 저장한 형식에 맞춘다
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, word: list[bytes]) -> list[bytes]:
        """사전 토큰 하나(bytes 리스트)에 병합을 만들어진 순서대로 적용 (p10 예제와 같음)"""
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            candidates = [p for p in pairs if p in self.rank]
            if not candidates:
                break
            best = min(candidates, key=lambda p: self.rank[p])
            new = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        return word

    def _encode_pretoken(self, pt: str) -> list[int]:
        if pt in self.cache:
            return self.cache[pt]
        word = [bytes([b]) for b in pt.encode("utf-8")]
        ids = [self.byte_to_id[tok] for tok in self._apply_merges(word)]
        self.cache[pt] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        # 1. 특수 토큰 기준으로 자른다 (특수 토큰도 결과 리스트에 남음)
        chunks = re.split(self.special_pat, text) if self.special_pat else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                # 특수 토큰은 쪼개지 않고 ID 하나로
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
                continue
            # 2. 일반 텍스트는 사전 토큰화 후 사전 토큰마다 병합 적용
            for m in re.finditer(PAT, chunk):
                ids.extend(self._encode_pretoken(m.group()))
        return ids

    def _safe_cut(self, buf: str) -> int:
        """buf의 앞부분 중 지금 인코딩해도 결과가 달라지지 않는 길이를 돌려준다. 0이면 아직 기다림"""
        # 1순위: 마지막 특수 토큰이 끝나는 곳. 특수 토큰은 어떤 병합도 넘지 못하는 경계라 항상 안전
        if self.special_pat:
            last = None
            for m in re.finditer(self.special_pat, buf):
                last = m
            if last is not None:
                return last.end()
        # 2순위: 특수 토큰 없이 버퍼가 너무 커지면, 마지막 사전 토큰 앞에서 자른다
        # (마지막 덩어리는 다음 입력과 이어질 수 있으니 남겨 둔다)
        if len(buf) > MAX_BUFFER:
            last = None
            for m in re.finditer(PAT, buf):
                last = m
            if last is not None:
                return last.start()
        return 0

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 줄 단위로 바로 인코딩하면 안 된다. 연속 줄바꿈('\n\n\n')이나 줄 끝 공백은
        # 정규식이 줄 경계를 넘어 한 덩어리로 묶기 때문에 결과가 달라진다.
        # 그래서 버퍼에 모아 두다가 안전한 경계까지만 인코딩한다.
        buf = ""
        for chunk in iterable:
            buf += chunk
            cut = self._safe_cut(buf)
            if cut > 0:
                yield from self.encode(buf[:cut])
                buf = buf[cut:]
        if buf:  # 입력이 끝나면 남은 것을 마저 인코딩
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        # 바이트를 전부 이어 붙인 뒤 한 번에 디코딩. 잘못된 바이트는 U+FFFD로
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")