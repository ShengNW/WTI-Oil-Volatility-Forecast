import pandas as pd
import numpy as np
from pathlib import Path

# 配置路径
project_root = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/04_docs/TestFormula/JWTSRV"#"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
data_path = Path(project_root) / '01_data/processed/intraday_returns'
data_path.mkdir(parents=True, exist_ok=True)

# 生成测试数据
dates = pd.date_range("2025-01-01 09:00", "2025-01-01 9:05", freq="1min")
np.random.seed(42)

# 基础收益率 (标准差=0.001)
returns = np.random.normal(0, 0.001, len(dates))

# 添加跳跃 (2025-01-02)
# jump_idx = [
#     np.where(dates == "2025-01-02 11:30")[0][0],
#     np.where(dates == "2025-01-02 14:45")[0][0]
# ]
# returns[jump_idx] = [0.025, -0.018]  # 显著跳跃

# 创建DataFrame
df = pd.DataFrame({
    "LogReturn": returns
}, index=dates)

# 移除第三天数据（模拟数据不足）
df = df[~df.index.normalize().isin(["2025-01-03"])]

# 保存为parquet
df.to_parquet(data_path / "CL_WTI_1min_processed.parquet")
print("测试数据生成完成")