import yfinance as yf
import pandas as pd
import os


def main():
    output_path = "data/nasdaq_5years_prices.csv"
    print("뉴스 데이터셋 타임라인에 맞춰 나스닥 주가 데이터 다운로드 중...")

    nasdaq = yf.Ticker("^IXIC")
    # 뉴스 아카이브의 실제 타임라인과 일치하도록 수집 기간 설정
    price_df = nasdaq.history(start="2010-01-01", end="2020-12-31")

    # 당일 등락률 변동성 계산
    price_df['Return'] = price_df['Close'].pct_change()
    price_df = price_df.reset_index()
    price_df['Date'] = price_df['Date'].dt.strftime('%Y-%m-%d')

    price_cleaned = price_df[['Date', 'Close', 'Return']].copy()
    price_cleaned = price_cleaned.dropna().reset_index(drop=True)

    print("\n=== 주가 데이터 확인 ===")
    print("수집 데이터 규모: ", len(price_cleaned), "일 거래일")

    os.makedirs("data", exist_ok=True)
    price_cleaned.to_csv(output_path, index=False)
    print(f"동기화 주가 파일이 {output_path} 경로에 저장되었습니다.")


if __name__ == "__main__":
    main()