from konlpy.tag import Komoran
import json
import jpype


class Preprocess:
    def __init__(self, dict_dir='', userDict_dir=None):
        # 단어 인덱스 사전 불러오기
        if(dict_dir != ''):
            with open(dict_dir, "r", encoding='utf-8') as f:
                self.word2idx = json.load(f) 
        else:
            self.word2idx = None

        # 형태소 분석기 초기화
        self.komoran = Komoran(userdic=userDict_dir)

        # 제외할 품사
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    # 형태소 분석기 POS 태거
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    # 불용어 제거 후, 필요한 품사 정보만 가져오기
    def get_keywords(self, pos, without_tag=True):
        f = lambda x: x in self.exclusion_tags
        words = []
        for p in pos:
            if f(p[1]) is False:
                words.append(p if without_tag is False else p[0])
        return words

    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_ids(self, keywords):
        if self.word2idx is None:
            return []

        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word2idx[word])
            except KeyError:
                # 해당 단어가 사전에 없는 경우, OOV 처리
                w2i.append(self.word2idx['OOV'])
        return w2i

