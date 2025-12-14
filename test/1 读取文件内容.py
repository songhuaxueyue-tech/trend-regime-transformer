# 导包
import pandas as pd


def read_file(file_path: str) -> pd.DataFrame:
    """
    读取指定路径的文件内容，支持 CSV 和 Excel 格式。
    
    参数:
        file_path (str): 文件的路径，支持 .csv 和 .xlsx 格式。
        
    返回:
        pd.DataFrame: 读取的文件内容作为 Pandas DataFrame。
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.feather') or file_path.endswith('.parquet'):
        df = pd.read_feather(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv, .feather, or .parquet files.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")


    return df


if __name__ == "__main__":
    # 示例用法
    file_path = "../data/BTC_USDT_USDT-15m-futures.feather"  # 替换为你的文件路径
    data = read_file(file_path)
    print(data.head(100))


