import pandas as pd
import openpyxl
from joblib import Parallel, delayed

# 读取数据
file_path = '105年A1-A4所有當事人.xlsx'
wb = openpyxl.load_workbook(file_path)
sheet = wb.worksheets[1]
data = sheet.values
columns = next(data)
df = pd.DataFrame(data, columns=columns)

# 使用 apply 和 map 進行操作
df = df.apply(lambda col: col.map(lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else x))


####创建新特征
new_df = pd.DataFrame()
#########刪除缺失數據
df= df.dropna(subset=['8道路型態']) #刪除此特徵沒數據的資料
# 要刪除的值列表
values_to_remove = [43, 44, 67]#原因不明代碼 對於分析結果無效
# 過濾掉包含這些值的行
df = df[~df['肇因碼-個別'].isin(values_to_remove)]

print(df)
# 定义函数
def categorize_day_night(day_night):
    return 0 if day_night == '夜' else 1

# 创建新特征
df['date'] = pd.to_datetime(df['發生時間'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['quarter'] = df['date'].dt.quarter
df['day_night_category'] = df['晝夜'].apply(categorize_day_night)

# 将案號列转换为相应的数字
df['案號'], _ = pd.factorize(df['案號'])

# 选择所需列并填充缺失值
cols_to_fill = ['4天候', '5光線', '6道路類別', '7速限', '8道路型態', '9事故位置', '10路面狀況1', 
                '10路面狀況2', '10路面狀況3', '11道路障礙1', '11道路障礙2', '12號誌1', 
                '12號誌2', '13車道劃分-分向', '14車道劃分-分道1', '14車道劃分-分道2', 
                '14車道劃分-分道3', '15事故類型及型態']
#df[cols_to_fill] = df[cols_to_fill].fillna(0)

# 创建新的 DataFrame
new_df = df[['year', 'month', 'day', 'hour', 'quarter', 'day_night_category', '案號', 'X', 'Y', '4天候', 
             '5光線', '6道路類別', '7速限', '8道路型態', '9事故位置', '10路面狀況1', '10路面狀況2', 
             '10路面狀況3', '11道路障礙1', '11道路障礙2', '12號誌1', '12號誌2', '13車道劃分-分向', 
             '14車道劃分-分道1', '14車道劃分-分道2', '14車道劃分-分道3', '15事故類型及型態', 
             '外國人', '性別', '肇因碼-個別']].dropna()

# 重命名列
new_df.columns = ['year', 'month', 'day', 'hour', 'quarter', 'day_night_category', 'case_id', 'X', 'Y', '4', 
                  '5', '6', '7', '8', '9', '101', '102', '103', '111', '112', '121', '122', '13', '141', 
                  '142', '143', '15', 'foreigner', 'gender', 'ans']

# 保存到 Excel 文件
output_file_path = 'output.xlsx'
new_df.to_excel(output_file_path, index=False)
