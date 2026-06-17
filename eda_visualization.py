"""
EDA 시각화 스크립트
- 원본 뉴스 헤드라인 문장 길이 분포
- 연도별 뉴스 수집 건수 분포
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import os

# 한글 폰트 설정
os_name = platform.system()
if os_name == 'Windows':
    font_family = 'Malgun Gothic'
elif os_name == 'Darwin':
    font_family = 'AppleGothic'
else:
    font_family = 'NanumBarunGothic'
sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

os.makedirs("output_images", exist_ok=True)


def save_text_length_distribution():
    """원본 뉴스 헤드라인 문장 길이 분포"""
    df = pd.read_csv("data/raw_news_50k.csv", encoding="utf-8-sig")

    # Full_Text 컬럼 확인
    text_col = None
    for col in ["Full_Text", "Headline_and_Subtitle", "headline", "text"]:
        if col in df.columns:
            text_col = col
            break

    df["text_length"] = df[text_col].astype(str).apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 히스토그램
    axes[0].hist(df["text_length"], bins=50, color="#005088", edgecolor="white", alpha=0.85)
    axes[0].axvline(df["text_length"].median(), color="#EF4444", linewidth=2,
                    linestyle="--", label=f'중간값: {df["text_length"].median():.0f}단어')
    axes[0].axvline(64, color="#FF7F0E", linewidth=2,
                    linestyle=":", label="max_length=64 기준선")
    axes[0].set_xlabel("헤드라인 단어 수 (Word Count)", fontsize=11)
    axes[0].set_ylabel("뉴스 건수", fontsize=11)
    axes[0].set_title("원본 뉴스 헤드라인 문장 길이 분포", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    # 오른쪽: 박스플롯
    axes[1].boxplot(df["text_length"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#005088", alpha=0.6),
                    medianprops=dict(color="#EF4444", linewidth=2))
    axes[1].set_ylabel("헤드라인 단어 수 (Word Count)", fontsize=11)
    axes[1].set_title("문장 길이 박스플롯", fontsize=13, fontweight="bold")
    axes[1].set_xticks([])

    # 통계 텍스트
    stats = df["text_length"].describe()
    stats_text = (f"평균: {stats['mean']:.1f}단어\n"
                  f"중간값: {stats['50%']:.0f}단어\n"
                  f"최솟값: {stats['min']:.0f}단어\n"
                  f"최댓값: {stats['max']:.0f}단어\n"
                  f"전체: {len(df):,}건")
    axes[1].text(1.3, stats['75%'], stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F0DF", edgecolor="#005088"))

    plt.suptitle("원본 뉴스 헤드라인 문장 길이 분석 (EDA)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output_images/eda_text_length.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("저장 완료: output_images/eda_text_length.png")


def save_yearly_distribution():
    """연도별 뉴스 수집 건수 분포"""
    df = pd.read_csv("data/raw_news_50k.csv", encoding="utf-8-sig")

    # 날짜 컬럼 확인
    date_col = "date" if "date" in df.columns else "Date"
    df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    yearly = df.groupby("year").size().reset_index(name="count")

    plt.figure(figsize=(12, 5))
    bars = plt.bar(yearly["year"], yearly["count"], color="#005088",
                   edgecolor="white", alpha=0.85, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 100,
                 f'{int(height):,}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    plt.xlabel("연도", fontsize=11)
    plt.ylabel("뉴스 수집 건수", fontsize=11)
    plt.title("연도별 원본 뉴스 수집 건수 분포", fontsize=13, fontweight="bold")
    plt.xticks(yearly["year"], rotation=45)
    plt.tight_layout()
    plt.savefig("output_images/eda_yearly_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("저장 완료: output_images/eda_yearly_distribution.png")


def save_train_text_length_distribution():
    """최종 학습 데이터(다운샘플링 완료)의 토큰 길이 분포"""
    from transformers import MobileBertTokenizer

    df = pd.read_csv("data/nasdaq_perfect_4_labels.csv", encoding="utf-8-sig")
    tokenizer = MobileBertTokenizer.from_pretrained("mobilebert-uncased")

    # 토큰 길이 계산 (special token 포함)
    token_lengths = df["Headline_and_Subtitle"].astype(str).apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True))
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 히스토그램
    axes[0].hist(token_lengths, bins=40, color="#11CAA0", edgecolor="white", alpha=0.85)
    axes[0].axvline(token_lengths.median(), color="#EF4444", linewidth=2,
                    linestyle="--", label=f'중간값: {token_lengths.median():.0f}토큰')
    axes[0].axvline(64, color="#FF7F0E", linewidth=2,
                    linestyle=":", label="max_length=64 기준선")
    axes[0].set_xlabel("토큰 길이 (Token Length)", fontsize=11)
    axes[0].set_ylabel("데이터 건수", fontsize=11)
    axes[0].set_title("학습 데이터 헤드라인 토큰 길이 분포", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    # 오른쪽: 64 토큰 초과 비율
    over_64 = (token_lengths > 64).sum()
    under_64 = (token_lengths <= 64).sum()
    pie_data = [under_64, over_64]
    pie_labels = [f"64 이하\n({under_64:,}건)", f"64 초과\n({over_64:,}건)"]
    axes[1].pie(pie_data, labels=pie_labels, colors=["#11CAA0", "#EF4444"],
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 11})
    axes[1].set_title("max_length=64 기준 충족 비율", fontsize=13, fontweight="bold")

    plt.suptitle("최종 학습 데이터(3,868건) 토큰 길이 분석", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output_images/eda_train_text_length.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("저장 완료: output_images/eda_train_text_length.png")
    print(f"  - 중간값: {token_lengths.median():.0f}토큰, 최댓값: {token_lengths.max()}토큰")
    print(f"  - 64 토큰 이하 비율: {under_64/len(token_lengths)*100:.1f}%")


if __name__ == "__main__":
    save_text_length_distribution()
    save_yearly_distribution()
    save_train_text_length_distribution()
    print("\n완료: output_images/ 폴더에 EDA 그래프 3개가 저장되었습니다.")