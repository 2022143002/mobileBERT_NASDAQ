from datasets import load_dataset
import pandas as pd
import os


def main():
    output_path = "data/raw_news_50k.csv"
    print("초대형 금융 뉴스 데이터셋(ashraq/financial-news) 엔진 가동 및 수집 개시...")

    try:
        # 185만 건 규모의 정통 월스트리트 금융 헤드라인 데이터셋 스트리밍 로드
        dataset = load_dataset("ashraq/financial-news", split="train", streaming=True)

        dates = []
        contents = []

        print("원격 서버로부터 순수 오가닉 금융 헤드라인 및 실제 발행 날짜 파싱 중...")
        for item in dataset:
            # 최종 결합 후 최소 25,000건 이상을 안전하게 유지하기 위해 60,000건 확보 시까지 수집
            if len(contents) >= 60000:
                break

            # 날짜 포맷 정제 (2020-06-01 00:00:00 -> YYYY-MM-DD 앞자리만 추출)
            raw_date = str(item.get('date', '')).strip()
            actual_date = raw_date[:10] if len(raw_date) >= 10 else ""

            headline = str(item.get('headline', '')).strip()

            if actual_date and headline:
                dates.append(actual_date)
                contents.append(headline)

        # 데이터 프레임 구축 및 중복 제거
        df = pd.DataFrame({
            'date': dates,
            'Full_Text': contents
        })
        df = df.drop_duplicates(subset=['Full_Text']).copy()

        print("\n=== 데이터 확인 ===")
        print("인위적 증강 없는 순수 금융 뉴스 수집 완료 개수: ", len(df), "개")

        # 결과 저장
        os.makedirs("data", exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"가짜 복제 없는 100% 진짜 금융 뉴스 파일이 {output_path} 경로에 저장되었습니다.")

    except Exception as e:
        print(f"\n데이터셋 로드 중 치명적 오류 발생: {e}")


if __name__ == "__main__":
    main()