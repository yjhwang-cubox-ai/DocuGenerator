import pandas as pd
import glob
from tqdm import tqdm

file_paths = glob.glob("./dataset/*.csv")

columns_to_read = [
    "상호", "사업자등록번호", "법인여부",
    "대표자명", "전화번호", "신고일자", "사업장소재지", "사업장소재지(도로명)"
]
# columns_to_read = [
#     "통신판매번호", "신고기관명", "상호", "사업자등록번호", "법인여부",
#     "대표자명", "전화번호", "신고일자", "사업장소재지", "사업장소재지(도로명)"
# ]

# 각 파일에서 데이터 읽기
df_list = []

# 사업자등록번호 패턴 정의 (XXX-XX-XXXXX 형식)
pattern = r'^\d{3}-\d{2}-\d{5}$'

for file in tqdm(file_paths):
    df_temp = pd.read_csv(file, on_bad_lines='skip')  # 파일 읽기
    
    # 사업자등록번호 형식이 맞는 데이터만 필터링
    mask = df_temp["사업자등록번호"].astype(str).str.match(pattern)
    filtered_df = df_temp[mask][columns_to_read]
    
    # 필요한 열만 선택하여 리스트에 추가
    df_list.append(filtered_df.dropna())

# 모든 데이터프레임을 하나로 합치기
df = pd.concat(df_list, ignore_index=True)  # 데이터프레임 합치기
df.to_csv('output.csv', index=False, encoding='utf-8-sig')
print(df)