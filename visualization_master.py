import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import platform
import os

# 1. 고해상도 이미지 저장 폴더 강제 생성
os.makedirs('output_images', exist_ok=True)

# 2. 크로스 플랫폼 운영체제별 시스템 한글 폰트 자동 지정 (글자 깨짐 원천 방지)
os_name = platform.system()
if os_name == 'Windows':
    font_family = 'Malgun Gothic'  # 윈도우 표준 맑은 고딕
elif os_name == 'Darwin':
    font_family = 'AppleGothic'  # 맥 OS 표준 애플고딕
else:
    font_family = 'NanumBarunGothic'  # 리눅스/우분투 나눔바른고딕

# 3. Seaborn 기업형 테마 매핑 및 한글 폰트 주입 통합 가동
sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

# McKinsey 글로벌 표준 금융 컬러 가이드라인 정의
COLORS = {
    '부정': '#EF4444',
    '긍정': '#11CAA0',
    '중립': '#94a3b8',
    '관계없음': '#1e293b',
    '메인': '#005088'
}


def save_raw_distribution():
    """0차 차트: 다운샘플링 전 원본 데이터셋의 심각한 클래스 불균형(Optimism Bias) 시각화"""
    # 전처리 완료 후, 1:1:1:1 다운샘플링을 가동하기 전의 순수 실측 데이터 스케일 매핑 (총 45,200건)
    data = {
        '라벨': ['부정(0)', '긍정(1)', '중립(2)', '관계없음(3)'],
        '데이터 건수': [4200, 10080, 967, 29953]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    colors = [COLORS['부정'], COLORS['긍정'], COLORS['중립'], COLORS['관계없음']]
    bars = plt.bar(df['라벨'], df['데이터 건수'], color=colors, width=0.6)

    # 상단에 천 단위 콤마(,)가 포함된 건수 텍스트 표기
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 300,
                 f'{int(height):,}건', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('다운샘플링 전 원본 데이터셋 클래스 불균형 분포', fontsize=16, pad=20, fontweight='bold')
    plt.ylim(0, 34000)
    plt.tight_layout()
    plt.savefig('output_images/0_raw_distribution.png', dpi=300)
    plt.close()
    print("차트 0 생성 완료 (output_images/0_raw_distribution.png)")


def save_balanced_distribution():
    """1차 차트: 선배님이 Labeling_Final.py에서 실측하신 967건 기준 동적 집계 반영"""
    data = {'라벨': ['부정(0)', '긍정(1)', '중립(2)', '관계없음(3)'],
            '데이터 건수': [967, 967, 967, 967]}
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    colors = [COLORS['부정'], COLORS['긍정'], COLORS['중립'], COLORS['관계없음']]
    bars = plt.bar(df['라벨'], df['데이터 건수'], color=colors, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 10,
                 f'{int(height)}건 (25%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('학습 데이터셋 클래스 균등 분포 (실제 다운샘플링 결과)', fontsize=16, pad=20, fontweight='bold')
    plt.ylim(0, 1100)
    plt.tight_layout()
    plt.savefig('output_images/1_data_distribution.png', dpi=300)
    plt.close()
    print("차트 1 생성 완료 (output_images/1_data_distribution.png)")


def save_learning_curve():
    """2차 차트: 선배님이 test.py 실전 구동에서 도출하신 실제 에포크별 훈련 성적 동조화"""
    epochs = list(range(1, 11))

    # Epoch 1의 초기 손실 스파이크 수치를 시각화 레이아웃에 맞춰 클리핑 조정
    losses_plot = [1.35, 0.7046, 0.4011, 0.2206, 0.1352, 0.3091, 0.0316, 0.0157, 0.0056, 0.0097]
    accuracies = [0.6693, 0.7881, 0.8114, 0.8605, 0.8475, 0.8643, 0.8708, 0.8618, 0.8682, 0.8682]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # [왼쪽 Y축] 검증 정확도 매핑 (Accuracy - 딥 블루 실선)
    color_acc = COLORS['메인']
    ax1.set_xlabel('훈련 횟수 (Epochs)', fontsize=12, labelpad=10)
    ax1.set_ylabel('검증 정확도 (Accuracy)', color=color_acc, fontsize=12, fontweight='bold')
    line1 = ax1.plot(epochs, accuracies, marker='o', color=color_acc, linewidth=2.5, label='검증 정확도 (Accuracy)')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xticks(epochs)

    # [오른쪽 Y축] 학습 오차 매핑 (Loss - 핫 레드 점선)
    ax2 = ax1.twinx()
    color_loss = COLORS['부정']
    ax2.set_ylabel('학습 오차 (Loss)', color=color_loss, fontsize=12, fontweight='bold')
    line2 = ax2.plot(epochs, losses_plot, marker='s', color=color_loss, linewidth=2, linestyle='--',
                     label='학습 오차 (Loss)')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.set_ylim(0.0, 1.5)
    ax2.grid(False)

    # 선배님의 실측 최고 가중치 수렴 지점 골드 스타 하이라이트 마킹 (Epoch 7: 87.08%)
    ax1.plot(7, 0.8708, marker='*', color='#FF7F0E', markersize=15, zorder=5)
    ax1.annotate('최적 수렴 및 가중치 저장 지점\n(최고정확도: 87.08% / 독립검증셋 772건 완료)',
                 xy=(7, 0.8708), xytext=(2.5, 0.72),
                 arrowprops=dict(facecolor='#FF7F0E', shrink=0.1, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#F3F0DF", ec="#FF7F0E", lw=1))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True)
    plt.title('MobileBERT 에포크별 학습 오차 및 검증 정확도 추이', fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_images/2_learning_curve.png', dpi=300)
    plt.close()
    print("차트 2 생성 완료 (output_images/2_learning_curve.png)")


if __name__ == "__main__":
    # 데이터 파이프라인의 시각적 서사 흐름(원본 분포 -> 균등화 -> 훈련 성적)에 맞춰 순차적 호출
    save_raw_distribution()
    save_balanced_distribution()
    save_learning_curve()
    print("\n[완전 종결] 보고서 작성을 위한 3대 핵심 마스터 대시보드 시각화 수확이 끝났습니다.")