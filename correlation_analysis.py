"""
결론용 Correlation 분석 스크립트 (라벨 직접 사용 버전)
- 모델 추론 없이 라벨링된 데이터를 날짜별로 집계
- 날짜별 N_p (긍정=1 건수) - N_n (부정=0 건수) 계산
- 실제 주가 등락 방향(0/1)과의 Pearson Correlation 산출
- 규칙 기반 라벨(Label)과 검수 후 라벨(Reviewed_Label) 두 가지 모두 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import platform
from scipy.stats import pearsonr


def analyze_correlation(df_news, df_price, label_col, label_name):
    df_news = df_news.copy()
    df_price = df_price.copy()

    if "date" in df_news.columns:
        df_news = df_news.rename(columns={"date": "Date"})
    df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    merged = pd.merge(df_news, df_price, on="Date", how="inner")
    print(f"\n[{label_name}] 뉴스-주가 매핑 완료: {len(merged):,}건")

    daily = merged.groupby("Date").apply(lambda x: pd.Series({
        "N_p": (x[label_col] == 1).sum(),
        "N_n": (x[label_col] == 0).sum(),
        "N_total": len(x),
        "Actual_Return": x["Return"].iloc[0]
    }), include_groups=False).reset_index()

    daily["Signal"] = daily["N_p"] - daily["N_n"]
    daily["Direction"] = (daily["Actual_Return"] > 0).astype(int)

    print(f"날짜 수: {len(daily)}일")
    print(daily[["Date", "N_p", "N_n", "Signal", "Actual_Return", "Direction"]].head(10).to_string(index=False))

    corr, p_value = pearsonr(daily["Signal"], daily["Direction"])

    print(f"\n=== [{label_name}] Pearson Correlation 결과 ===")
    print(f"Correlation (r) : {corr:.4f}")
    print(f"P-value         : {p_value:.4f}")

    if abs(corr) >= 0.7:
        interpretation = "유의미한 상관관계 (r >= 0.7)"
    elif abs(corr) > 0.5:
        interpretation = "약한 상관관계 (0.5 < r < 0.7, 큰 의미 없음)"
    else:
        interpretation = "상관관계 없음 (r <= 0.5)"
    print(f"해석            : {interpretation}")

    return daily, corr, p_value, interpretation


def main():
    os_name = platform.system()
    if os_name == 'Windows':
        font_family = 'Malgun Gothic'
    elif os_name == 'Darwin':
        font_family = 'AppleGothic'
    else:
        font_family = 'NanumBarunGothic'
    sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

    news_path = "data/nasdaq_perfect_4_labels.csv"
    price_path = "data/nasdaq_5years_prices.csv"
    manual_path = "data/manual_review_samples.csv"

    if not os.path.exists(news_path) or not os.path.exists(price_path):
        print("오류: 데이터 파일이 없습니다.")
        return

    df_news = pd.read_csv(news_path, encoding="utf-8-sig")
    df_price = pd.read_csv(price_path)

    # 1. 규칙 기반 라벨 분석
    print("=" * 60)
    print("1. 규칙 기반 라벨 (Label) 상관관계 분석")
    print("=" * 60)
    daily_rule, corr_rule, pval_rule, interp_rule = analyze_correlation(
        df_news, df_price, label_col="Label", label_name="규칙 기반"
    )

    # 2. 검수 후 라벨 분석
    daily_reviewed = None
    corr_reviewed = None
    pval_reviewed = None
    interp_reviewed = None

    if os.path.exists(manual_path):
        manual_df = pd.read_csv(manual_path, encoding="utf-8-sig")

        df_news_reviewed = df_news.copy()
        df_news_reviewed["Reviewed_Label"] = df_news_reviewed["Label"].copy()

        if "date" in df_news_reviewed.columns:
            df_news_reviewed = df_news_reviewed.rename(columns={"date": "Date"})
        df_news_reviewed["Date"] = pd.to_datetime(df_news_reviewed["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

        changed = 0
        for _, row in manual_df.iterrows():
            mask = df_news_reviewed["Headline_and_Subtitle"] == row["Headline_and_Subtitle"]
            if mask.any():
                df_news_reviewed.loc[mask, "Reviewed_Label"] = int(row["Manual_Label"])
                if int(row["Label"]) != int(row["Manual_Label"]):
                    changed += 1
        print(f"\n검수 반영: {changed}건 수정")

        print("\n" + "=" * 60)
        print("2. 검수 후 라벨 (Reviewed_Label) 상관관계 분석")
        print("=" * 60)
        daily_reviewed, corr_reviewed, pval_reviewed, interp_reviewed = analyze_correlation(
            df_news_reviewed, df_price, label_col="Reviewed_Label", label_name="검수 후"
        )

    # 결과 요약
    print("\n" + "=" * 60)
    print("=== 최종 결과 요약 ===")
    print("=" * 60)
    print(f"{'데이터 기준':<15} | {'Pearson r':>10} | {'p-value':>10} | 해석")
    print("-" * 65)
    print(f"{'규칙 기반 라벨':<15} | {corr_rule:>10.4f} | {pval_rule:>10.4f} | {interp_rule}")
    if corr_reviewed is not None:
        print(f"{'검수 후 라벨':<15} | {corr_reviewed:>10.4f} | {pval_reviewed:>10.4f} | {interp_reviewed}")

    # 시각화
    os.makedirs("output_images", exist_ok=True)

    if corr_reviewed is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        datasets = [
            (daily_rule, corr_rule, pval_rule, "규칙 기반 라벨", axes[0]),
            (daily_reviewed, corr_reviewed, pval_reviewed, "검수 후 라벨", axes[1])
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        datasets = [(daily_rule, corr_rule, pval_rule, "규칙 기반 라벨", ax)]

    for daily, corr, pval, name, ax in datasets:
        colors = daily["Direction"].map({1: "#11CAA0", 0: "#EF4444"})
        ax.scatter(daily["Signal"], daily["Actual_Return"] * 100,
                   c=colors, alpha=0.55, edgecolors="none", s=35)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("감성 신호 (N_p - N_n)", fontsize=11)
        ax.set_ylabel("실제 나스닥 등락률 (%)", fontsize=11)
        ax.set_title(f"[{name}]\n감성 신호 vs 실제 등락률\n(r = {corr:.4f}, p = {pval:.4f})",
                     fontsize=11, fontweight="bold")
        legend_elements = [
            mpatches.Patch(facecolor="#11CAA0", label="실제 상승"),
            mpatches.Patch(facecolor="#EF4444", label="실제 하락")
        ]
        ax.legend(handles=legend_elements, fontsize=10)

    plt.suptitle("뉴스 감성 신호(N_p - N_n)와 나스닥 주가 변동 방향 상관관계 분석",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output_images/4_correlation_analysis.png", dpi=300, bbox_inches="tight")
    print("\n차트 저장 완료: output_images/4_correlation_analysis.png")

    daily_rule.to_csv("data/correlation_result_rule.csv", index=False, encoding="utf-8-sig")
    if daily_reviewed is not None:
        daily_reviewed.to_csv("data/correlation_result_reviewed.csv", index=False, encoding="utf-8-sig")
    print("날짜별 집계 결과 저장 완료")

    plt.show()


if __name__ == "__main__":
    main()