import os
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 한글 폰트 및 마이너스 기호 깨짐 방지
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 깔끔한 화이트그리드 테마 설정
    sns.set_theme(style="whitegrid", font="Malgun Gothic")

    # 리포트에 적힌 에포크별 실측 데이터 입력
    epochs = list(range(1, 11))
    losses = [1.2543, 0.7776, 0.7432, 0.4972, 0.1432, 0.0982, 0.1602, 0.0314, 0.0124, 0.0058]
    accuracies = [0.6486, 0.7868, 0.8204, 0.8463, 0.8514, 0.8527, 0.8514, 0.8747, 0.8656, 0.8630]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # 1. 왼쪽 축: 검증 정확도 (Accuracy - 파란색 실선)
    color_acc = '#1F77B4'
    ax1.set_xlabel('훈련 횟수 (Epoch)', fontsize=12, labelpad=10)
    ax1.set_ylabel('검증 정확도 (Accuracy)', color=color_acc, fontsize=12)
    line1 = ax1.plot(epochs, accuracies, marker='o', color=color_acc, linewidth=2.5, label='검증 정확도 (Accuracy)')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xticks(epochs)

    # 2. 오른쪽 축: 학습 오차 (Loss - 빨간색 점선)
    ax2 = ax1.twinx()
    color_loss = '#D62728'
    ax2.set_ylabel('학습 오차 (Loss)', color=color_loss, fontsize=12)
    line2 = ax2.plot(epochs, losses, marker='s', color=color_loss, linewidth=2, linestyle='--', label='학습 오차 (Loss)')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    ax2.set_ylim(0.0, 1.5)
    ax2.grid(False)  # 격자 중첩 방지

    # 에포크 8 최고점 하이라이트 마킹 및 가이드라인
    ax1.axvline(x=8, color='#FF7F0E', linestyle=':', linewidth=2)
    ax1.plot(8, 0.8747, marker='*', color='#FF7F0E', markersize=15, zorder=5)
    ax1.text(8, 0.89, '최적 가중치 보존 지점\n(Epoch 8: 87.47%)',
             ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')

    # 범례 합치기
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)

    plt.title('MobileBERT 에포크별 학습 오차 및 검증 정확도 추이 (Learning Curve)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # 이미지 저장
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/learning_curve.png", dpi=300, bbox_inches='tight')
    print("[성공] 학습 곡선 그래프가 'images/learning_curve.png'로 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    main()