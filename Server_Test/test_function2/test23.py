import FinanceDataReader as fdr
import pandas as pd
import  json
import random
import csv
def main():
    # 오늘의 AI 랜덤값을 만든다.
    # CSV 파일 읽기
    csv_filename = '../stock_list.csv'  # 파일 이름을 적절하게 변경해주세요
    data_df = pd.read_csv(csv_filename)

    target_codes = []
    target_names = []

    while len(target_names) < 5:
        # 랜덤한 행 번호 생성 (중복 없이)
        random_number = random.choice(range(1, len(data_df) + 1))

        # 선택된 행 번호로부터 데이터 가져오기
        selected_row = data_df.iloc[random_number - 1]

        # 'Code' 열의 값이 target_codes와 일치하면 다시 랜덤한 행 번호를 선택
        if selected_row['Code'] in target_codes:
            continue

        target_names.append(selected_row['Name'])
        target_codes.append(selected_row['Code'])

    today_ai_csv_filename = 'today_ai_list.csv'
    # CSV 파일 생성 및 데이터 쓰기
    with open(today_ai_csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Code', 'Name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # 헤더(필드명) 쓰기

        for code, name in zip(target_codes, target_names):
            writer.writerow({'Code': code, 'Name': name})


main()