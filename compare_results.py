"""
[Step 3] 규칙 기반 라벨 vs 검수 후 라벨 학습 결과 비교 시각화
- test.py 실행 결과(규칙 기반)와 step2_train_reviewed.py 실행 결과(검수 후)를 비교
- 두 학습 곡선을 하나의 그래프에 나란히 출력
- 결과를 output_images/3_comparison.png 로 저장
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import os


def main():
    # 한글 폰트
    os_name = platform.system()
    if os_name == 'Windows':
        font_family = 'Malgun Gothic'
    elif os_name == 'Darwin':
        font_family = 'AppleGothic'
    else:
        font_family = 'NanumBarunGothic'
    sns.set_theme(style="whitegrid", rc={"font.family": font_family, "axes.unicode_minus": False})

    # ── 규칙 기반 실측 결과 (test.py 실행 결과 직접 입력) ──────────────────
    rule_results = {
        "Epoch":      [1,      2,      3,      4,      5,      6,      7,      8,      9,      10],
        "Loss":       [1.2543, 0.7776, 0.7432, 0.4972, 0.1432, 0.0982, 0.1602, 0.0314, 0.0124, 0.0058],
        "Train_Acc":  [0.6486, 0.7868, 0.8204, 0.8463, 0.8514, 0.8527, 0.8514, 0.8747, 0.8656, 0.8630],
        "Valid_Acc":  [0.6693, 0.7881, 0.8114, 0.8605, 0.8475, 0.8643, 0.8708, 0.8618, 0.8682, 0.8682],
    }
    df_rule = pd.DataFrame(rule_results)

    # ── 검수 후 결과 (step2 실행 후 자동 저장된 CSV 로드) ──────────────────
    reviewed_path = "data/reviewed_training_results.csv"
    if not os.path.exists(reviewed_path):
        print(f"오류: {reviewed_path} 파일이 없습니다. step2_train_reviewed.py를 먼저 실행하세요.")
        return

    df_reviewed = pd.read_csv(reviewed_path, encoding="utf-8-sig")
    df_reviewed["Epoch"] = df_reviewed["Epoch"]  # index → 컬럼

    epochs = df_rule["Epoch"].tolist()

    # ── 핵심 수치 비교 출력 ──────────────────────────────────────────────────
    best_rule = df_rule["Valid_Acc"].max()
    best_rule_epoch = df_rule.loc[df_rule["Valid_Acc"].idxmax(), "Epoch"]
    best_reviewed = df_reviewed["Valid_Acc"].max()
    best_reviewed_epoch = df_reviewed.loc[df_reviewed["Valid_Acc"].idxmax(), "Epoch"]
    diff = best_reviewed - best_rule

    print("="*55)
    print("=== 규칙 기반 vs 검수 후 라벨 학습 결과 비교 ===")
    print("="*55)
    print(f"{'구분':<12} | {'최고 검증 정확도':>14} | {'달성 Epoch':>10}")
    print("-"*45)
    print(f"{'규칙 기반':<12} | {best_rule:>14.4f} | {best_rule_epoch:>10}")
    print(f"{'검수 후':<12} | {best_reviewed:>14.4f} | {best_reviewed_epoch:>10}")
    print("-"*45)

    if diff > 0.005:
        verdict = f"개선  (+{diff*100:.2f}%p)"
    elif diff < -0.005:
        verdict = f"저하  ({diff*100:.2f}%p)"
    else:
        verdict = f"변화없음  ({diff*100:.2f}%p)"
    print(f"결과: {verdict}")
    print("="*55)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: 검증 정확도 비교
    axes[0].plot(epochs, df_rule["Valid_Acc"], marker='o', linewidth=2.5,
                 color='#005088', label='규칙 기반 라벨')
    axes[0].plot(df_reviewed["Epoch"], df_reviewed["Valid_Acc"], marker='s', linewidth=2.5,
                 color='#11CAA0', linestyle='--', label='검수 후 라벨')
    axes[0].axvline(x=best_rule_epoch, color='#005088', linestyle=':', linewidth=1.2, alpha=0.6)
    axes[0].axvline(x=best_reviewed_epoch, color='#11CAA0', linestyle=':', linewidth=1.2, alpha=0.6)
    axes[0].set_title('검증 정확도 비교 (Valid Accuracy)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('검증 정확도', fontsize=11)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_xticks(epochs)
    axes[0].legend(fontsize=10)
    axes[0].annotate(f'규칙 기반\n최고: {best_rule:.4f}',
                     xy=(best_rule_epoch, best_rule),
                     xytext=(best_rule_epoch + 0.5, best_rule - 0.05),
                     fontsize=9, color='#005088',
                     arrowprops=dict(arrowstyle='->', color='#005088', lw=1.2))
    axes[0].annotate(f'검수 후\n최고: {best_reviewed:.4f}',
                     xy=(best_reviewed_epoch, best_reviewed),
                     xytext=(best_reviewed_epoch - 2.5, best_reviewed + 0.03),
                     fontsize=9, color='#11CAA0',
                     arrowprops=dict(arrowstyle='->', color='#11CAA0', lw=1.2))

    # 오른쪽: 학습 오차 비교
    axes[1].plot(epochs, df_rule["Loss"], marker='o', linewidth=2.5,
                 color='#EF4444', label='규칙 기반 라벨')
    axes[1].plot(df_reviewed["Epoch"], df_reviewed["Loss"], marker='s', linewidth=2.5,
                 color='#F97316', linestyle='--', label='검수 후 라벨')
    axes[1].set_title('학습 오차 비교 (Loss)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('학습 오차', fontsize=11)
    axes[1].set_xticks(epochs)
    axes[1].legend(fontsize=10)

    # 최고 정확도 비교 텍스트 박스
    verdict_text = f"최고 검증 정확도\n규칙 기반: {best_rule:.4f} (Epoch {best_rule_epoch})\n검수 후:   {best_reviewed:.4f} (Epoch {best_reviewed_epoch})\n결과: {verdict}"
    fig.text(0.5, -0.04, verdict_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3F0DF', edgecolor='#FF7F0E', lw=1.2))

    plt.suptitle('MobileBERT 라벨 품질별 학습 성능 비교\n(규칙 기반 라벨 vs 검수 후 라벨)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs("output_images", exist_ok=True)
    plt.savefig("output_images/3_comparison.png", dpi=300, bbox_inches='tight')
    print("\n비교 그래프 저장 완료: output_images/3_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()