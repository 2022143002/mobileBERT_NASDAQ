import pandas as pd
import numpy as np
import os


def determine_smart_label(headline):
    """
    [고도화 버전] 사전을 보강하여 데이터 확보량을 늘린 라벨링 엔진
    """
    text = str(headline).lower().strip()

    # 규칙 1: 가짜 뉴스 및 단순 공시 차단
    blog_keywords = [
        'will', 'how to', 'should you', 'analyst blog', 'column',
        'what to look out for', 'guide', 'preview', 'outlook', 'strategy',
        'history', 'recession generation', 'worried about', 'is it time',
        'transcript', 'conference call', 'earnings call presentation'
    ]
    if any(word in text for word in blog_keywords):
        return 3

        # 규칙 2: 문맥 반전 구제 (Phrase Modifiers)
    if 'risk' in text:
        risk_controls = ['control', 'manage', 'reduce', 'hedge', 'protect', 'mitigate']
        if any(ctrl in text for ctrl in risk_controls):
            return 2

    if 'loss' in text or 'losses' in text:
        loss_reductions = ['cut', 'cuts', 'reduce', 'reduces', 'lower', 'lowers', 'narrow', 'narrows']
        if any(red in text for red in loss_reductions):
            return 1

    reverse_negatives = ['give back gains', 'gains fall short', 'erase gains', 'misses on revenue', 'earnings miss']
    if any(rev in text for rev in reverse_negatives):
        return 0

    strong_positives = ['great pick', 'top pick', 'strong value', 'best pick', 'worthy of catching']
    if any(sp in text for sp in strong_positives):
        return 1

    # 규칙 3: 보강된 Loughran-McDonald 사전 단어 (데이터 확보량 증대용)
    pos_words = [
        'gain', 'beat', 'rise', 'growth', 'surge', 'profit', 'outpace', 'higher',
        'surpass', 'outshine', 'soar', 'up', 'dividend', 'investment', 'acquire',
        'acquisition', 'boost', 'expand', 'success', 'record', 'high'
    ]
    neg_words = [
        'drop', 'miss', 'loss', 'tumble', 'fall', 'decline', 'slump', 'plummet',
        'warning', 'downgrade', 'hurt', 'bad', 'crumble', 'plunge', 'tariff',
        'debt', 'deficit', 'shudder', 'worsen', 'low', 'weakness', 'cut', 'pressure'
    ]

    pos_count = sum(1 for word in pos_words if word in text)
    neg_count = sum(1 for word in neg_words if word in text)

    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return 0
    elif pos_count == 0 and neg_count == 0:
        return 3
    else:
        return 2


def main():
    raw_path = "data/raw_news_50k.csv"
    output_path = "data/nasdaq_perfect_4_labels.csv"

    if not os.path.exists(raw_path):
        print(f"오류: {raw_path} 파일이 없습니다.")
        return

    print("=== 1단계: 원본 데이터 로드 ===")
    df = pd.read_csv(raw_path)

    print("\n=== 2단계: 보강된 스마트 라벨링 규칙 적용 ===")
    target_column = None
    possible_columns = ['Headline_and_Subtitle', 'Headline', 'title', 'text', 'Text', 'headline', 'Full_Text']

    for col in possible_columns:
        if col in df.columns:
            target_column = col
            break

    if target_column is None:
        print("오류: 뉴스 컬럼을 찾지 못했습니다.")
        return

    df['Label'] = df[target_column].apply(determine_smart_label)
    if target_column != 'Headline_and_Subtitle':
        df = df.rename(columns={target_column: 'Headline_and_Subtitle'})

    print(df['Label'].value_counts())

    print("\n=== 3단계: 최신 문법 기반 균등 다운샘플링 ===")
    min_class_size = df['Label'].value_counts().min()
    print(f"방당 확보 데이터 개수: {min_class_size}건")

    # FutureWarning 해결을 위한 최신 sample 문법
    balanced_df = df.groupby('Label').sample(n=min_class_size, random_state=2026).reset_index(drop=True)

    print("\n=== 4단계: 데이터셋 저장 ===")
    balanced_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")
    print(balanced_df['Label'].value_counts())


if __name__ == "__main__":
    main()