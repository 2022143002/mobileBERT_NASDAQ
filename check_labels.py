import pandas as pd
import os
from deep_translator import GoogleTranslator


def main():
    data_path = "data/nasdaq_perfect_4_labels.csv"
    price_path = "data/nasdaq_5years_prices.csv"  # 주가 동기화 파일 경로 추가

    if not os.path.exists(data_path):
        print("오류: 최후 가공된 4채널 라벨링 파일이 존재하지 않습니다.")
        return

    df = pd.read_csv(data_path)

    # [정합성 패치] Labeling_Final 스크립트 출력물에 없는 주가 데이터를 실시간 조인하여 보정함
    if os.path.exists(price_path):
        df_price = pd.read_csv(price_path)

        # 훈련 표준 규격에 맞춰 날짜 컬럼명 및 포맷 동기화
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_price['Date'] = pd.to_datetime(df_price['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # 날짜 기준으로 실제 주가 등락률 수익률 결합 (Inner Join)
        df = pd.merge(df, df_price, on='Date', how='inner')
        df['Market_Return'] = df['Return']
    else:
        print("경고: 주가 데이터 파일이 없어 변동률 지표가 0으로 대체됩니다.")
        df['Market_Return'] = 0.0
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})

    # 확인하고 싶은 라벨별 샘플 추출 개수 설정
    samples_per_label = 2

    # 새로운 라벨 규격 정의 (0:악재, 1:호재, 2:중립, 3:관계없음)
    label_names = {
        1: "긍정 (1: 호재성 시황)",
        0: "부정 (0: 악재성 시황)",
        2: "중립 (2: 보합권 뉴스)",
        3: "관계없음 (3: 금융 외 뉴스)"
    }

    print("=== 4채널 자동 라벨링 정성적 대조 검증 개시 (구글 실시간 번역 탑재) ===")

    for label_code in [1, 0, 2, 3]:
        sub_df = df[df['Label'] == label_code]

        print(f"\n==================================================")
        print(f"■ 검증 타겟 채널: {label_names[label_code]}")
        print(f"==================================================")

        # 실전 런타임 결과 수치인 967건 기반의 동조화 메시지 출력
        print(f"전체 파일 내 해당 채널 데이터 수: {len(sub_df)}개")

        if len(sub_df) == 0:
            print("해당 라벨로 분류된 데이터가 없습니다.")
            continue

        n_extract = min(samples_per_label, len(sub_df))
        sample_data = sub_df.sample(n=n_extract, random_state=42)

        for idx, (_, row) in enumerate(sample_data.iterrows(), start=1):
            eng_text = row['Headline_and_Subtitle']

            # 구글 번역 엔진을 가동하여 영문 뉴스를 한글로 실시간 변환한다
            try:
                kor_text = GoogleTranslator(source='en', target='ko').translate(eng_text)
            except Exception:
                kor_text = "번역 엔진 통신 오버헤드 발생으로 한글 변환 실패"

            print(f"\n[{idx}번째 추출 샘플]")
            print(f"거래 일자 : {row['Date']}")
            print(f"영문 원문 : {eng_text}")
            print(f"한글 번역 : {kor_text}")
            print(f"당일 나스닥 변동률 : {row['Market_Return']:.4f} ({row['Market_Return'] * 100:.2f}%)")
            print(f"부여된 라벨 코드 : {row['Label']}")
            print(f"-" * 40)


if __name__ == "__main__":
    main()