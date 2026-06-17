"""
결론용 Correlation 분석 스크립트
- test.py로 학습된 모델을 활용하여 전체 데이터에 대해 감성 추론
- 날짜별 N_p - N_n 계산 (긍정 예측 수 - 부정 예측 수)
- 실제 주가 등락 방향(0/1)과의 Pearson Correlation 산출
- 상관계수 해석 기준:
    r >= 0.7        → 유의미한 상관관계
    0.5 < r < 0.7  → 약한 상관관계 (큰 의미 없음)
    r <= 0.5       → 상관관계 없음
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import platform
from tqdm import tqdm
from scipy.stats import pearsonr


def main():
    # 한글 폰트 설정
    os_name = platform.system()
    if os_name == 'Windows':
        font_family = 'Malgun Gothic'
    elif os_name == 'Darwin':
        font_family = 'AppleGothic'
    else:
        font_family = 'NanumBarunGothic'
    sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 디바이스:", device)

    # 1. 모델 로드
    model_path = "./best_mobilebert_model_reviewed"
    if not os.path.exists(model_path):
        print("오류: 학습 완료된 모델이 없습니다. test.py를 먼저 실행하세요.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # 2. 데이터 로드
    news_path = "data/nasdaq_perfect_4_labels.csv"
    price_path = "data/nasdaq_5years_prices.csv"

    if not os.path.exists(news_path) or not os.path.exists(price_path):
        print("오류: 데이터 파일이 없습니다.")
        return

    df_news = pd.read_csv(news_path, encoding="utf-8-sig")
    df_price = pd.read_csv(price_path)

    # 날짜 컬럼명 및 포맷 통일
    if "date" in df_news.columns:
        df_news = df_news.rename(columns={"date": "Date"})
    df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # 뉴스 + 주가 Inner Join (실제 거래일 기준)
    merged = pd.merge(df_news, df_price, on="Date", how="inner")
    print(f"뉴스-주가 매핑 완료: {len(merged):,}건 (실제 거래일 기준)")

    # 3. MobileBERT 감성 추론
    print("\n=== MobileBERT 감성 분류 추론 시작 ===")
    texts = merged["Headline_and_Subtitle"].astype(str).tolist()
    predicted_labels = []

    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="추론 중"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predicted_labels.extend(preds)

    merged = merged.copy()
    merged["Predicted_Label"] = [int(p) for p in predicted_labels]

    print("\n[예측 라벨 분포]")
    print(merged["Predicted_Label"].value_counts().sort_index())

    # 4. 날짜별 N_p, N_n, Signal 계산
    print("\n=== 날짜별 감성 신호 집계 ===")
    daily = merged.groupby("Date").apply(lambda x: pd.Series({
        "N_p": (x["Predicted_Label"] == 1).sum(),   # 긍정 예측 건수
        "N_n": (x["Predicted_Label"] == 0).sum(),   # 부정 예측 건수
        "N_total": len(x),
        "Actual_Return": x["Return"].iloc[0]
    }), include_groups=False).reset_index()

    # Signal = N_p - N_n (양수: 긍정 우세, 음수: 부정 우세)
    daily["Signal"] = daily["N_p"] - daily["N_n"]

    # 실제 주가 방향 이진화 (상승=1, 하락=0)
    daily["Direction"] = (daily["Actual_Return"] > 0).astype(int)

    print(f"날짜 수: {len(daily)}일")
    print(daily[["Date", "N_p", "N_n", "Signal", "Actual_Return", "Direction"]].head(10).to_string(index=False))

    # 5. Pearson Correlation 계산
    corr, p_value = pearsonr(daily["Signal"], daily["Direction"])

    print(f"\n=== Pearson Correlation 결과 ===")
    print(f"Correlation (r) : {corr:.4f}")
    print(f"P-value         : {p_value:.4f}")

    if abs(corr) >= 0.7:
        interpretation = "유의미한 상관관계 (r >= 0.7)"
    elif abs(corr) > 0.5:
        interpretation = "약한 상관관계 (0.5 < r < 0.7, 큰 의미 없음)"
    else:
        interpretation = "상관관계 없음 (r <= 0.5)"
    print(f"해석            : {interpretation}")

    # 6. 시각화
    os.makedirs("output_images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 차트 1: Signal vs 실제 등락률 산점도
    colors = daily["Direction"].map({1: "#11CAA0", 0: "#EF4444"})
    axes[0].scatter(daily["Signal"], daily["Actual_Return"] * 100,
                    c=colors, alpha=0.55, edgecolors="none", s=35)
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].axvline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("감성 신호 (N_p - N_n)", fontsize=11)
    axes[0].set_ylabel("실제 나스닥 등락률 (%)", fontsize=11)
    axes[0].set_title(f"감성 신호 vs 실제 등락률\n(r = {corr:.4f},  p = {p_value:.4f})",
                      fontsize=12, fontweight="bold")
    legend_elements = [
        mpatches.Patch(facecolor="#11CAA0", label="실제 상승"),
        mpatches.Patch(facecolor="#EF4444", label="실제 하락")
    ]
    axes[0].legend(handles=legend_elements, fontsize=10)

    # 차트 2: Signal 방향별 실제 주가 상승 비율
    daily["Signal_Direction"] = daily["Signal"].apply(
        lambda x: "긍정 우세\n(N_p > N_n)" if x > 0
        else ("부정 우세\n(N_n > N_p)" if x < 0 else "동률")
    )
    hit_rate = daily.groupby("Signal_Direction")["Direction"].mean().reset_index()
    hit_rate.columns = ["Signal_Direction", "상승비율"]

    bar_colors = {
        "긍정 우세\n(N_p > N_n)": "#11CAA0",
        "부정 우세\n(N_n > N_p)": "#EF4444",
        "동률": "#94a3b8"
    }
    bar_c = [bar_colors.get(x, "#94a3b8") for x in hit_rate["Signal_Direction"]]
    bars = axes[1].bar(hit_rate["Signal_Direction"], hit_rate["상승비율"] * 100,
                       color=bar_c, width=0.5, edgecolor="gray", alpha=0.85)
    axes[1].axhline(50, color="gray", linewidth=1, linestyle="--", label="기준선 (50%)")
    axes[1].set_ylabel("실제 나스닥 상승 비율 (%)", fontsize=11)
    axes[1].set_title("감성 신호 방향별 실제 주가 상승 비율", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=10)

    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1.5,
                     f"{height:.1f}%", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    plt.suptitle(
        f"뉴스 감성 신호(N_p - N_n)와 나스닥 주가 변동 방향 상관관계 분석\n"
        f"Pearson r = {corr:.4f}  |  {interpretation}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("output_images/4_correlation_analysis.png", dpi=300, bbox_inches="tight")
    print("\n차트 저장 완료: output_images/4_correlation_analysis.png")

    # 7. 날짜별 집계 결과 저장
    daily.to_csv("data/correlation_result.csv", index=False, encoding="utf-8-sig")
    print("날짜별 집계 결과 저장: data/correlation_result.csv")

    plt.show()


if __name__ == "__main__":
    main()