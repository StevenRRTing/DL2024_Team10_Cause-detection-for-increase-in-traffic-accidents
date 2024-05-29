import pandas as pd
import openpyxl
##1. read data
# 使用 openpyxl 打開 Excel 文件
file_path = '109年A1-A4所有當事人(新增戶籍地).xlsx'

# 使用 openpyxl 打開 Excel 文件
wb = openpyxl.load_workbook(file_path)


# 選擇第二個工作表
sheet = wb.worksheets[1]

# 將數據加載到 DataFrame
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
new_df = df[~df['肇因碼-個別'].isin(values_to_remove)]



##2.time
# 定义一个函数，根据小时判断是白天还是夜晚，返回0代表夜晚，1代表早上
def categorize_day_night(day_night):
    if day_night == '夜':
        return 0  # 夜晚
    else:
        return 1  # 白天


df['date'] = pd.to_datetime(df['發生時間'])# 将 '發生時間' 列转换为日期时间格式
new_df ['year'] = df['date'].dt.year.copy()
new_df ['month'] = df['date'].dt.month  # 修改此处，将月份赋值给 'month'
new_df ['day'] = df['date'].dt.day
new_df ['hour'] = df['date'].dt.hour
new_df ['quarter'] = df['date'].dt.quarter
new_df['day_night_category'] = df['晝夜'].apply(lambda x: categorize_day_night(x))# 应用函数到 '晝夜' 列



##3.place
numeric_case_numbers, _ = pd.factorize(df['案號'])# 將案號列轉換為相應的數字
# 將轉換後的數字列添加到 DataFrame 中
df['案號'] = numeric_case_numbers
new_df['case_id'] = df['案號']
new_df ['X'] = df['X']
new_df ['Y'] = df['Y']
new_df['4'] = df['4天候']
new_df['5'] = df['5光線']
new_df['6'] = df['6道路類別']
new_df['7'] = df['7速限']
new_df['8'] = df['8道路型態']
new_df['9'] = df['9事故位置']
new_df['101'] = df['10路面狀況1']
new_df['102'] = df['10路面狀況2']
new_df['103'] = df['10路面狀況3']
new_df['111'] = df['11道路障礙1']
new_df['112'] = df['11道路障礙2']
new_df['121'] = df['12號誌1']
new_df['122'] = df['12號誌2']
new_df['13'] = df['13車道劃分-分向']
new_df['141'] = df['14車道劃分-分道1']
new_df['142'] = df['14車道劃分-分道2']
new_df['143'] = df['14車道劃分-分道3']
new_df['15'] = df['15事故類型及型態']


##4.people
new_df['foreigner'] = df['外國人']
#new_df['military_vehicle'] = df['軍車']
#new_df['police'] = df['員警']
#new_df['major_cases'] = df['重大案件']
#new_df['major_vehicle_damage'] = df['重大車損']
#new_df['blood_test'] = df['抽血']
#new_df['hit_and_run'] = df['肇逃']
#new_df['after_report'] = df['事後報案']
#new_df['report_date'] = df['事後報案日']
#new_df['failure_to_comply'] = df['未依規定處置']
#new_df['non_traffic_accident'] = df['非交通事故']
#new_df['nationality'] = df['國籍']
new_df['gender'] = df['性別']
#new_df['22'] = df['22受傷程度']
#new_df['23'] = df['23主要傷處']
#new_df['24'] = df['24安全帽']
#new_df['25'] = df['25行動電話']
#new_df['28'] = df['28車輛用途']
#new_df['29'] = df['29當事者行動狀態']
#new_df['30'] = df['30駕駛資格情形']
#new_df['31'] = df['31駕駛執照種類']
#new_df['32'] = df['32飲酒情形']
#new_df['33_1'] = df['33_1主要車損']
#new_df['33_2'] = df['33_2其他車損']
#new_df['35'] = df['35個人肇逃否']
#new_df['36'] = df['36職業']
#new_df['37'] = df['37旅次目的']
##output
#new_df['ans'] = df['肇因碼-個別']
#new_df['original'] = new_df['ans'].isnull().astype(int)

#new_df = new_df.dropna(thresh=len(new_df) * 0.5, axis=1)
#new_df = new_df.dropna(thresh=len(new_df.columns) * 0.7, axis=1)

new_df = new_df.dropna()
output_file_path = '109.xlsx'
new_df.to_excel(output_file_path, index=False)
