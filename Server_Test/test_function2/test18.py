import os
import pandas as pd
import json

# CSV 파일이 들어 있는 디렉토리 경로
directory_path = './models/today/'  # 실제 디렉토리 경로로 변경해주세요

# 디렉토리 내의 파일 목록 조회
file_list = os.listdir(directory_path)

# 확장자가 '.csv'이고 파일명이 '035720'로 시작하는 파일 찾기
matching_files = [filename for filename in file_list if filename.startswith('000100') and filename.endswith('.csv')]
# 첫 번째 매칭 파일 선택 (해당 디렉토리 내에 하나의 파일만 있어야 함)
if len(matching_files) == 1:
    selected_filename = matching_files[0]

    # CSV 파일 읽기
    csv_filepath = os.path.join(directory_path, selected_filename)
    data_df = pd.read_csv(csv_filepath)

    # 데이터를 원하는 형식의 JSON으로 변환
    json_data = {}
    for index, row in data_df.iterrows():
        json_data[row['Date']] = row['Price']

    # JSON 데이터 출력 또는 파일로 저장
    print(json.dumps(json_data, indent=4))
else:
    print("해당 조건을 만족하는 파일이 없거나 여러 개입니다.")