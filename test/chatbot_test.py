import sys 
sys.path.append("../")

import json

import torch
from tensorflow.keras import preprocessing
from konlpy.tag import Komoran

from utils.Database import Database
from utils.Preprocess import Preprocess
from utils.FindAnswer import FindAnswer

from config.DatabaseConfig import DB
from config.IntentConfig import MODEL as intentConfig
from config.NerConfig import MODEL as nerConfig

from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel 


def main():
    # DB 생성 및 연결
    db = Database(DB['db_host'],
                DB['db_user'],
                DB['db_password'],
                DB['db_name'],
                'utf8')
    db.connect()

    idx2intent = {0: "인사",
                  1: "욕설",
                  2: "주문",
                  3: "예약",
                  4: "기타"}
    
    idx2ner = {1: 'O',
               2: 'B_DT',
               3: 'B_FOOD',
               4: 'I',
               5: 'B_OG',
               6: 'B_PS',
               7: 'B_LC',
               8: 'NNP',
               9: 'B_TI',
               0: 'PAD'}

    query = "오전에 탕수육 10개 주문합니다."

    intentModel = IntentModel(intentConfig)
    nerModel = NerModel(nerConfig)

    intent_checkpoint = torch.load('../models/intent/best.tar', 
                                   map_location=torch.device('cpu'))
    ner_checkpoint = torch.load('../models/ner/best.tar', 
                               map_location=torch.device('cpu'))

    intentModel.load_state_dict(intent_checkpoint['model_state_dict'])
    nerModel.load_state_dict(ner_checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    intentModel.to(device)
    nerModel.to(device)
    
    # 전처리 
    p = Preprocess(dict_dir='../train_tools/dict/chatbot_dict.json',
                   userDict_dir='../utils/user_dic.tsv')
    query_pos = p.pos(query)

    intentInput = p.get_keywords(query_pos)
    intentInput = p.get_ids(intentInput)
    intentInput = preprocessing.sequence.pad_sequences([intentInput],
                                                       padding='post',
                                                       value=0,
                                                       maxlen=intentConfig['maxlen'])
    intentInput = torch.tensor(intentInput).to(device)

    nerInput = [pos[0] for pos in query_pos]
    nerInput = p.get_ids(nerInput)
    nerInput = preprocessing.sequence.pad_sequences([nerInput],
                                                    padding='post',
                                                    value=0,
                                                    maxlen=nerConfig['maxlen'])
    nerInput = torch.tensor(nerInput).to(device)

    # 예측
    pred_intent = intentModel(intentInput)
    pred_ner = nerModel(nerInput)

    pred_intent = pred_intent.max(dim=-1)[1].to('cpu').tolist()[0]
    pred_ner = pred_ner.max(dim=-1)[1].to('cpu').tolist()[0]

    intent = idx2intent[pred_intent]
    tags = []
    for tag in pred_ner:
        if tag == 1 or tag == 0: 
            continue
        tags.append(idx2ner[tag])

    print("query: ", query)
    print("=" * 40)
    print("intent: ", intent)
    print("ner: ", pred_ner)
    print("tag: ", tags)
    print("="*40)

    try:
        f = FindAnswer(db)
        answer_text, answer_image = f.search(intent, tags)
    except:
        answer_text = "죄송해요, 무슨 말인지 모르겠어요"
    
    print("answer: ", answer_text)
    db.close()

if __name__=='__main__':
    main()

        
    


    




