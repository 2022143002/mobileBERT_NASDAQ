import pandas as pd
import os
from tqdm import tqdm
# [추가] 가장 안정적인 무료 구글 번역 라이브러리 인입
from deep_translator import GoogleTranslator


def main():
    path = "data/nasdaq_perfect_4_labels.csv"
    output_path = "data/human_audit_2000.csv"

    if not os.path.exists(path):
        print("오류: 가공된 라벨링 파일이 없습니다.")
        return

    df = pd.read_csv(path, encoding="utf-8-sig")

    # 각 라벨(0, 1, 2, 3)에서 정확히 500개씩 샘플링
    audit_df = df.groupby("Label").apply(
        lambda x: x.sample(n=500, random_state=42) if len(x) >= 500 else x
    ).reset_index(drop=True)

    # --- [선배님 지시 반영: 자동 한국어 번역 로직] ---
    print("\n=== 영어 뉴스 2,000건 한국어 번역 개시 (약 2~3분 소요) ===")
    translator = GoogleTranslator(source='en', target='ko')

    korean_texts = []
    # 2,000개 문장을 하나씩 번역하면서 진행 상황을 게이지(tqdm)로 보여줍니다.
    for text in tqdm(audit_df['Headline_and_Subtitle'].tolist(), desc="한국어로 번역 중"):
        try:
            if pd.isna(text) or str(text).strip() == "":
                korean_texts.append("")
            else:
                # 구글 서버를 통해 안전하게 번역 수행
                translated = translator.translate(str(text))
                korean_texts.append(translated)
        except Exception as e:
            # 네트워크 일시 오류 등으로 실패할 경우 프로그램을 멈추지 않고 원문을 남겨둡니다.
            korean_texts.append(f"[번역실패] {text}")

    # 번역된 리스트를 새로운 컬럼으로 삽입
    audit_df['Headline_and_Subtitle_KO'] = korean_texts
    # --------------------------------------------------

    # 인간이 눈으로 보기 편하게 컬럼 순서 정렬 (영어 원문 바로 옆에 한국어 번역본 배치!)
    audit_df = audit_df[['Date', 'Label', 'Headline_and_Subtitle', 'Headline_and_Subtitle_KO', 'Market_Return']]

    # 저장
    audit_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n=== 인간 검증용 샘플 2,000건 추출 및 번역 완료 ===")
    print(audit_df['Label'].value_counts())
    print(f"파일 경로: {output_path} (이 파일을 엑셀로 열어 편하게 한글로 확인하세요!)")


if __name__ == "__main__":
    main()