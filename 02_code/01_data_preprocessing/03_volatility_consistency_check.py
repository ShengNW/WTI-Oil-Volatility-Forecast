import pandas as pd
import numpy as np
from pathlib import Path
import os
import pyarrow.parquet as pq
import warnings

warnings.filterwarnings('ignore')


def check_volatility_consistency():
    # 设置文件路径
    input_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates")
    output_report = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates/consistency_report.csv")

    # 确保输出目录存在
    output_report.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有parquet文件
    parquet_files = list(input_dir.glob("*.parquet"))

    # 初始化报告数据
    report_data = []

    # 定义预期的日期范围
    expected_start = pd.Timestamp("2015-01-05")
    expected_end = pd.Timestamp("2024-12-30")
    expected_rows = 2475

    for file in parquet_files:
        file_info = {
            "filename": file.name,
            "datetime_index": "Fail",
            "date_range": "Fail",
            "row_count": "Fail",
            "G_value": "N/A",
            "RK_kernel": "N/A",
            "MedRV_logic": "N/A",
            "issues": ""
        }

        try:
            # 读取文件
            df = pd.read_parquet(file)
            original_columns = df.columns.tolist()

            # 1. 检查并统一时间索引格式
            datetime_index_issue = False

            # 情况1: 索引已经是DateTimeIndex (带或不带时区)
            if isinstance(df.index, pd.DatetimeIndex):
                # 检查时区信息
                if df.index.tz is not None:
                    # 移除时区信息以统一格式
                    df.index = df.index.tz_localize(None)
                    datetime_index_issue = True
                    file_info["issues"] += "Removed timezone from index; "

                # 确保索引名称统一为"DateTime"
                if df.index.name != "DateTime":
                    df.index.name = "DateTime"
                    datetime_index_issue = True
                    file_info["issues"] += "Renamed index to 'DateTime'; "

            # 情况2: 索引不是DateTimeIndex，但有DateTime列
            elif "DateTime" in df.columns:
                # 设置DateTime为索引并移除原列
                df = df.set_index("DateTime")
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                datetime_index_issue = True
                file_info["issues"] += "Set 'DateTime' column as index; "

            # 情况3: 有Date列（如realized_kernel_estimates文件）
            elif "Date" in df.columns:
                # 转换Date列为DateTimeIndex
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                df.index.name = "DateTime"
                datetime_index_issue = True
                file_info["issues"] += "Converted 'Date' column to DateTime index; "

            # 情况4: 没有日期列，但索引可以转换为日期
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "DateTime"
                    datetime_index_issue = True
                    file_info["issues"] += "Converted index to DateTime; "
                except:
                    file_info["issues"] += "No valid date index or column found; "
                    file_info["datetime_index"] = "Fail"
                    report_data.append(file_info)
                    continue

            # 检查索引类型
            if isinstance(df.index, pd.DatetimeIndex):
                file_info["datetime_index"] = "Pass"
                if datetime_index_issue:
                    file_info["issues"] += "Fixed datetime index; "
            else:
                file_info["datetime_index"] = "Fail"
                file_info["issues"] += "Could not create datetime index; "

            # 2. 验证时间范围和行数
            if file_info["datetime_index"] == "Pass":
                # 检查行数
                row_count = len(df)
                file_info["row_count"] = f"{row_count}/{expected_rows}"

                # 检查日期范围
                date_range = f"{df.index.min().date()} to {df.index.max().date()}"
                if df.index.min() == expected_start and df.index.max() == expected_end:
                    file_info["date_range"] = "Pass"
                else:
                    file_info["date_range"] = "Fail"
                    file_info["issues"] += f"Date range mismatch: {date_range}; "

                # 检查行数是否符合预期
                if row_count != expected_rows:
                    file_info["issues"] += f"Row count mismatch: expected {expected_rows}, got {row_count}; "

            # 3. 验证特定波动率指标参数
            # TSRV文件验证 (G=16)
            if "TSRV" in file.name:
                if "G" in df.columns:
                    if (df["G"] == 16).all():
                        file_info["G_value"] = "Pass"
                    else:
                        file_info["G_value"] = "Fail"
                        file_info["issues"] += "G value not consistently 16; "
                else:
                    file_info["G_value"] = "Fail"
                    file_info["issues"] += "Missing G column; "

            # Realized Kernel文件验证
            if file.name == "realized_kernel_estimates.parquet":
                # 验证列存在性
                if "RK_5min" in df.columns and "RK_1min" in df.columns:
                    file_info["RK_kernel"] = "Pass (columns present)"
                else:
                    file_info["RK_kernel"] = "Fail (missing columns)"
                    file_info["issues"] += "Missing RK columns; "

            # MedRV文件验证
            if "MedRV" in file.name:
                # 基本逻辑检查：值应为正数
                if "MedRV" in df.columns:
                    if (df["MedRV"] >= 0).all():
                        file_info["MedRV_logic"] = "Pass (non-negative)"
                    else:
                        file_info["MedRV_logic"] = "Fail (negative values)"
                        file_info["issues"] += "MedRV contains negative values; "
                else:
                    file_info["MedRV_logic"] = "Fail (missing column)"
                    file_info["issues"] += "Missing MedRV column; "

            # 保存修复后的文件（如果有修改）
            if datetime_index_issue or file_info["issues"]:
                df.to_parquet(file)
                file_info["issues"] += "Saved corrected file; "

            # 清理issues字符串
            if file_info["issues"].endswith("; "):
                file_info["issues"] = file_info["issues"][:-2]

        except Exception as e:
            file_info["issues"] += f"Error processing file: {str(e)}"

        report_data.append(file_info)

    # 生成报告
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_report, index=False)
    print(f"Consistency report generated at: {output_report}")
    print("\nConsistency Check Summary:")
    print(report_df[["filename", "datetime_index", "date_range", "row_count"]])

    # 返回报告数据用于检查
    return report_df


if __name__ == "__main__":
    report = check_volatility_consistency()