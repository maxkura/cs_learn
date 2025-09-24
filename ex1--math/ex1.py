import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
import pandas as pd

class mirror():
    def __init__(self, x, y):
        self.x=x
        self.y=y
        self.z=0
        self.title=None

    #def shadow(self):

# 指定 Excel 文件路径（修改为你的实际路径）
excel_path = "/home/maxkura/exercise/fujian.xlsx"  # 也支持 .xls 格式

try:
    # 1. 读取整个 Excel 文件
    # sheet_name=None 读取所有工作表，返回字典类型
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    #print(all_sheets)
    print(f"成功读取文件! 包含 {len(all_sheets)} 个工作表")
    print("="*50)

    sheet0=all_sheets['Sheet1']
    x=sheet0['x坐标 (m)']
    y=sheet0['y坐标 (m)']
#pandas to numpy
    xx=x.values
    yy=y.values
    x2=np.power(xx,2)
    y2=np.power(yy,2)
    d=np.power(np.add(x2,y2),0.5)
    r_value= np.zeros(18, dtype=int)
#提取距离信息
    j=0
    r_value[0]=int(d[0])
    for i in range(len(d)-1):
        if r_value[j]!=int(d[i+1]):
           r_value[j+1]=int(d[i+1])
           j+=1
    print(r_value)
except FileNotFoundError:
    print(f"错误: 文件 {excel_path} 未找到! 请检查路径")
#对所有点进行初始化并分组
title={'n':[] for n in range(len(r_value))}
flag=0

for jj in range(len(x)):
    if r_value[flag]==int(d[jj]):    
        title[f'{flag}'].append()=mirror(x[jj],y[jj])
    else:
        flag+=1
        title[f'{flag}'].append()=mirror(x[jj],y[jj])
print(title)
print('sort is end')

    
        




        

    







#画出位置图片
    # plt.scatter(x, y, s=1)
    # plt.xlabel('x坐标 (m)')
    # plt.ylabel('y坐标 (m)')
    # plt.title('散点图')
    # plt.show()

#对所有点进行分组



#将每个点生成一个对象
#     # 2. 逐工作表处理数据
#     for sheet_name, df in all_sheets.items():
#         print(f"\n工作表名称: [{sheet_name}]")
#         print(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        
#         # 显示前3行数据
#         print("\n数据预览 (前3行):")
#         print(df.head(3))
        
#         # 显示列名和数据类型
#         print("\n列信息:")
#         for idx, col in enumerate(df.columns):
#             dtype = str(df.dtypes[idx]).replace("dtype('", "").replace("')", "")
#             print(f"  {idx+1}. [{col}] : {dtype}")
        
#         print("-"*30)

#     # 3. 获取特定工作表
#     first_sheet_name = list(all_sheets.keys())[0]
#     df_main = pd.read_excel(excel_path, sheet_name=first_sheet_name)
    
#     # 4. 基本数据分析示例
#     print("\n" + "="*50)
#     print(f"工作表 [{first_sheet_name}] 的统计摘要:")
#     print(df_main.describe())
    
# except FileNotFoundError:
#     print(f"错误: 文件 {excel_path} 未找到! 请检查路径")
# except Exception as e:
#     print(f"发生错误: {str(e)}")
# finally:
#     print("\n程序执行完成")
