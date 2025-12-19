import pandas as pd
import os
import sys

def check_data_structure():
    # 定义相对于脚本运行位置的文件路径
    # 假设脚本在项目根目录下运行
    file_path = os.path.join('meta_bidding', 'data', 'aemo_data', 'aemo_price_train_dual.pkl')
    
    # 获取绝对路径以便调试
    abs_path = os.path.abspath(file_path)
    print(f"正在尝试读取文件: {abs_path}")
    
    if not os.path.exists(file_path):
        print(f"\n[错误] 文件不存在: {file_path}")
        
        # 尝试列出目录内容，帮助用户排查
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            print(f"\n目录 '{dir_path}' 下的文件有:")
            files = os.listdir(dir_path)
            for f in files:
                print(f" - {f}")
        else:
            print(f"\n目录 '{dir_path}' 也不存在。")
        return

    try:
        df = pd.read_pickle(file_path)
        
        print("\n" + "="*30)
        print("数据加载成功！")
        print("="*30)
        
        print(f"\n数据形状 (Rows, Columns): {df.shape}")
        
        print("\n=== 列名列表 (Columns) ===")
        # 分组打印列名，方便查看
        da_cols = [c for c in df.columns if c.startswith('DA_')]
        rt_cols = [c for c in df.columns if not c.startswith('DA_') and c != 'SETTLEMENTDATE' and c != 'REGIONID']
        other_cols = [c for c in df.columns if c not in da_cols and c not in rt_cols]
        
        print(f"\n[日前市场列 (DA_*) - 共 {len(da_cols)} 个]:")
        for col in da_cols:
            print(f"  - {col}")
            
        print(f"\n[实时市场列 (RT) - 共 {len(rt_cols)} 个]:")
        for col in rt_cols:
            print(f"  - {col}")
            
        print(f"\n[其他列 - 共 {len(other_cols)} 个]:")
        for col in other_cols:
            print(f"  - {col}")

        print("\n=== 数据预览 (前 5 行) ===")
        pd.set_option('display.max_columns', None) # 显示所有列
        pd.set_option('display.width', 1000)
        print(df.head())
        
        print("\n=== 基本统计信息 ===")
        print(df.describe())

    except Exception as e:
        print(f"\n[异常] 读取或处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_structure()
