import json

from flask import jsonify
import fasttext
import sentencepiece as spm


def cb_fast_text(input):
    indexList = [[9, 7, 37, 6, 5, 17, 14, 20, 18, 37],
                 [3, 32, 38, 25, 33, 42, 15, 26, 49, 39, 24, 29],
                 [4, 36, 21, 45, 8, 11, 16, 43, 40, 28, 35, 1, 50, 10],
                 [51, 27, 41, 19, 48, 23, 22, 2],
                 [56, 55, 31, 46, 54, 12, 13, 44, 30, 34, 52, 53, 57]
                 ]
    categoryList = ['패션/뷰티', '가전/컴퓨터', '가구/생활/건강', '식품/유아동', '여행/레저/자동차']

    categoryClass = {
        "여성의류": 9,
        "남성의류": 7,
        "테마의류/잡화": 47,
        "스포츠의류/운동화/잡화": 6,
        "언더웨어": 5,
        "신발/수제화": 17,
        "가방/지갑/잡화": 14,
        "쥬얼리/시계/액세서리": 20,
        "스킨케어/메이크업": 18,
        "향수/바디/헤어": 37,

        "휴대폰/액세서리": 3,
        "디카/캠코더/주변기기": 32,
        "영상가전/TV/홈시어터": 38,
        "음향가전/스피커/전자사전": 25,
        "생활가전/세탁기/청소기": 33,
        "주방가전/냉장고/전기밥솥": 42,
        "계절가전/에어컨/온열기기": 15,
        "이미용/건강/욕실가전": 26,
        "노트북/태블릿PC": 49,
        "데스크탑/모니터/PC부품": 39,
        "프린터/PC주변/사무기기": 24,
        "게임/주변기기": 29,

        "침실가구": 4,
        "거실/주방가구": 36,
        "수납/정리/선반": 21,
        "홈오피스/키즈가구": 45,
        "침구/커튼/카페트": 8,
        "홈/인테리어/가드닝": 11,
        "주방/식기/용기": 16,
        "생활/제지/잡화": 43,
        "욕실/청소/세제": 40,
        "산업/공구/안전용품": 28,
        "문구/사무/용지": 35,
        "악기/취미/만들기": 1,
        "반려동물/애완용품": 50,
        "건강관리/실버용품": 10,

        "쌀/과일/농축수산물": 51,
        "가공식품/과자/초콜릿": 27,
        "음료/생수/커피": 41,
        "홍삼/건강/다이어트식품": 19,
        "분유/기저귀/물티슈": 48,
        "출산/유아용품/임부복": 23,
        "유아동의류/신발/가방": 22,
        "완구/교육/교구": 2,

        "해외여행": 56,
        "국내여행": 55,
        "여행": 31,
        "공연": 46,
        "상품권/e쿠폰/서비스": 54,
        "등산/캠핑/낚시": 12,
        "구기/헬스/수영/스키": 13,
        "자전거/인라인/모터사이클": 44,
        "골프클럽/의류/용품": 30,
        "자동차용품": 34,
        "내비/블랙박스/하이패스": 52,
        "도서/음반/DVD": 53,
        "성인": 57,
    }

    reverse_dict = dict(map(reversed, categoryClass.items()))

    quantized_model = fasttext.load_model('chocoBread_fastText_model.ftz')
    sp = spm.SentencePieceProcessor()
    sp.Load("spm.model")

    token_string = (' ').join(sp.EncodeAsPieces(input))
    print("Token String : " + token_string)
    result = quantized_model.predict(token_string, k=3)
    result_big_category = []
    result_small_category = []
    result_prob = result[1]
    for i in range(len(result[0])):
        index = int(result[0][i].strip("__label__"))
        result_small_category.append(reverse_dict.get(index))
        for j in range(len(indexList)):
            if index in indexList[j]:
                result_big_category.append(categoryList[j])
                break
    json_result = []
    for i in range(3):
        if result_prob[i] > 0.1 :
            print(result_big_category[i] + " > " + result_small_category[i] + " | " + str(result_prob[i]))
            json_result.append({"BIG" : result_big_category[i], "SMALL" : result_small_category[i], "prob" : result_prob[i]})
    return json.dumps(json_result, ensure_ascii=False, indent=4)
