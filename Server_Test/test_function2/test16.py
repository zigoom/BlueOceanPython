import pandas as pd
import random

# CSV 파일 읽기
csv_filename = 'stock_list.csv'  # 파일 이름을 적절하게 변경해주세요
data_df = pd.read_csv(csv_filename)

target_codes = [35720, 102281, 323411]
selected_rows = []

while len(selected_rows) < 5:
    # 랜덤한 행 번호 생성 (중복 없이)
    random_number = random.choice(range(1, len(data_df) + 1))

    # 선택된 행 번호로부터 데이터 가져오기
    selected_row = data_df.iloc[random_number - 1]

    # 'Code' 열의 값이 target_codes와 일치하면 다시 랜덤한 행 번호를 선택
    if selected_row['Code'] in target_codes:
        continue

    selected_rows.append(selected_row)
    print(f"Selected Row {len(selected_rows)}:")
    print(selected_row)
