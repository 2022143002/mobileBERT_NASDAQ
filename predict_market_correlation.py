import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    # 0. OS별 시스템 한글 폰트 지정 및 장치 설정
    os_name = platform.system()
    if os_name == 'Windows':
        font_family = 'Malgun Gothic'
    elif os_name == 'Darwin':
        font_family = 'AppleGothic'
    else:
        font_family = 'NanumBarunGothic'

    # Seaborn 테마 설정과 한글 폰트 주입 통합 가동 (글자 깨짐 방지 오버라이딩)
    sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

    # [정정 패치] torch.available() 구문 오타를 torch.cuda.is_available() 표준 명령으로 전면 수정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("현재 활성화된 연산 디바이스 백엔드 : ", device)

    # 1. 가중치 로드
    model_path = "./best_mobilebert_model"
    if not os.path.exists(model_path):
        print("오류: 학습 완료된 모델 폴더가 없습니다. test.py를 먼저 실행하세요.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 2. 뉴스 데이터 및 수집된 실제 주가 데이터 로드
    news_data_path = "data/nasdaq_perfect_4_labels.csv"
    price_data_path = "data/nasdaq_5years_prices.csv"

    if not os.path.exists(news_data_path) or not os.path.exists(price_data_path):
        print("오류: 데이터 파일이 누락되었습니다. 경로를 확인하세요.")
        return

    df_news = pd.read_csv(news_data_path)
    df_price = pd.read_csv(price_data_path)

    # 뉴스 데이터의 날짜 컬럼명을 주가 데이터의 'Date'와 일치하도록 통일
    if "date" in df_news.columns:
        df_news = df_news.rename(columns={"date": "Date"})

    # 날짜 포맷을 문자열 형식(%Y-%m-%d)으로 완벽하게 일치시켜 매칭율 극대화
    df_news["Date"] = pd.to_datetime(df_news["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors='coerce').dt.strftime('%Y-%m-%d')

    # [정합성 패치] test.py와 100% 동일한 검증셋 분리를 위해 뉴스 데이터 단독 조건에서 먼저 분할 가동
    _, val_news_df = train_test_split(df_news, test_size=0.2, random_state=2026, stratify=df_news["Label"])

    # 분리 완료된 순수 독립 검증셋(772건 풀)에 한해서만 실제 나스닥 주가 데이터를 날짜 기준으로 교차 병합 (Inner Join)
    val_df = pd.merge(val_news_df, df_price, on="Date", how="inner")
    val_df["Market_Return"] = val_df["Return"]

    print(f"=== 검증 데이터 동기화 완료: 실제 거래일 기준 {len(val_df):,}건의 뉴스-주가 쌍 확보 ===")

    # 선배님의 로깅 데이터 컬럼명 규격인 'Headline_and_Subtitle' 강제 고정 매핑
    texts = val_df["Headline_and_Subtitle"].astype(str).tolist()
    actual_returns = val_df["Market_Return"].values

    # 3. 모델 기반 뉴스 기사 전체 감성 분류(추론) 개시
    print("\n=== MobileBERT 기반 검증 데이터셋 전수 추론 및 분류 개시 ===")
    predicted_labels = []

    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predicted_labels.extend(preds)

    # 4. 분석 데이터 프레임 조립
    analysis_df = pd.DataFrame({
        "Predicted_Label": predicted_labels,
        "Actual_Return": actual_returns
    })

    id2label = {0: "부정 (악재 판단)", 1: "긍정 (호재 판단)", 2: "중립 (보합 판단)", 3: "관계없음 (금융 외)"}
    analysis_df["Predicted_Name"] = analysis_df["Predicted_Label"].map(id2label)

    # 5. 핵심 통계 연산 (아웃라이어 왜곡 방지를 위해 mean 대신 median 산출)
    summary = analysis_df.groupby("Predicted_Name")["Actual_Return"].agg(["count", "median"]).reset_index()
    summary["median_percent"] = summary["median"] * 100

    print("\n=== 뉴스 감성 분류별 실제 나스닥 주가 변동률 대조 결과 (중간값 기준) ===")
    print(summary.to_string(index=False))

    # 6. Matplotlib/Seaborn 통합 시각화 보고서 생성
    plt.figure(figsize=(10, 6))

    COLORS = {
        '부정 (악재 판단)': '#EF4444',
        '긍정 (호재 판단)': '#11CAA0',
        '중립 (보합 판단)': '#94a3b8',
        '관계없음 (금융 외)': '#1e293b'
    }
    colors = [COLORS.get(name, 'lightgray') for name in summary["Predicted_Name"]]

    bars = plt.bar(summary["Predicted_Name"], summary["median_percent"], color=colors, edgecolor="gray", alpha=0.8,
                   width=0.5)

    plt.title("뉴스 기사 감성 분류별 실제 나스닥 등락률 중간값(Median) 검증", fontsize=14, pad=20, fontweight='bold')
    plt.ylabel("실제 나스닥 등락률 중간값 (%)", fontsize=11, labelpad=10)
    plt.xlabel("MobileBERT 모델의 뉴스 기사 감성 분류 결과", fontsize=11, labelpad=10)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        va_side = 'bottom' if height >= 0 else 'top'
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}%',
                 ha='center', va=va_side, fontsize=10, fontweight='bold')

    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/market_correlation_result.png", dpi=300)
    print("\n[성공] 실제 주가 기반 중간값 분석 그래프가 'images/market_correlation_result.png'에 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    main()