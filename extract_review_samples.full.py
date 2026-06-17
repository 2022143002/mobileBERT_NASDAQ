"""
전체 학습 데이터 직접 판단용 추출 스크립트
- 학습 데이터 전체(3,868건)를 한국어 번역 포함하여 CSV로 저장
- 저장된 파일을 엑셀로 열어 전체 Manual_Label 컬럼을 직접 검토 및 수정
- 완료 후 test_reviewed.py 실행하여 재학습
- 번역 대상이 많아(3,868건) 약 15~20분 소요됩니다.
"""
import pandas as pd
import os
from tqdm import tqdm
from deep_translator import GoogleTranslator

LABELING_GUIDE = """
[직접 판단 라벨링 기준]
0 (부정/악재) : 기업 실적 악화, 주가 하락, 경기 침체, 파산, 소송, 제재 등 시장에 부정적 영향이 명확한 기사
               예) "Dow plunges 500 points", "Company files for bankruptcy"

1 (긍정/호재) : 실적 상회, 주가 상승, 인수합병 성공, 신제품 출시, 금리 인하 등 시장에 긍정적 영향이 명확한 기사
               예) "Apple beats earnings", "Fed cuts rates"

2 (중립)      : 긍정/부정이 혼재하거나 방향성이 불분명, 단순 수치 발표
               예) "GDP grows 2.1%, slightly below forecast"

3 (관계없음)  : 주가와 직접 관련 없는 기사, 전망/예측 칼럼, 투자 가이드, 기업 IR 공시
               예) "How to invest in ETFs", "Earnings call transcript"

[애매한 경우 판단 우선순위]
1. 수치 + 명확한 상승/하락 표현 → 긍정(1) 또는 부정(0)
2. but / however / despite 등 역접 표현 → 중립(2)
3. 금융 관련이지만 방향성 없음 → 중립(2)
4. 투자 조언 / 전망 / 가이드성 → 관계없음(3)
"""


def main():
    input_path = "data/nasdaq_perfect_4_labels.csv"
    output_path = "data/manual_review_samples.csv"

    if not os.path.exists(input_path):
        print(f"오류: {input_path} 파일이 없습니다. NASDAQ_Labeling_Final.py를 먼저 실행하세요.")
        return

    print(LABELING_GUIDE)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"전체 학습 데이터 {len(df):,}건 전수 검토용 파일을 생성합니다.")
    print("(번역 작업에 약 15~20분 소요될 수 있습니다)")

    # 전체 데이터 사용 (샘플링 없음)
    sample_df = df.copy().reset_index(drop=True)

    # 한국어 번역
    print("\n=== 한국어 번역 중 ===")
    translator = GoogleTranslator(source='en', target='ko')
    korean_texts = []

    for text in tqdm(sample_df["Headline_and_Subtitle"].tolist(), desc="번역 중"):
        try:
            if pd.isna(text) or str(text).strip() == "":
                korean_texts.append("")
            else:
                korean_texts.append(translator.translate(str(text)))
        except Exception:
            korean_texts.append("[번역 실패]")

    sample_df["Headline_KO"] = korean_texts
    sample_df["Manual_Label"] = sample_df["Label"]  # 기본값: 규칙 기반 라벨
    sample_df["수정여부"] = ""                        # 수정했으면 'Y' 입력

    # 컬럼 순서 정렬
    base_cols = ["Label", "Manual_Label", "수정여부", "Headline_and_Subtitle", "Headline_KO"]
    optional_cols = ["Date", "date", "Market_Return", "Return"]
    existing_optional = [c for c in optional_cols if c in sample_df.columns]
    result_df = sample_df[existing_optional + base_cols].copy()

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 전체 직접 판단용 데이터 {len(result_df):,}건 추출 완료 ===")
    print(f"저장 경로: {output_path}")
    print("\n[안내] CSV를 엑셀로 열어 'Manual_Label' 컬럼을 전체 검토 및 수정하고,")
    print("       수정한 항목의 '수정여부' 컬럼에 Y를 입력한 뒤 저장하세요.")
    print("       완료 후 test_reviewed.py를 실행하세요.")


if __name__ == "__main__":
    main()