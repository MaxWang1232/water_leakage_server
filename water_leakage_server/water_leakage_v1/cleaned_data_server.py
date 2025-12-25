import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from filelock import FileLock

logger = logging.getLogger(__name__)

def clean_water_data(raw_path, cleaned_path, lock, is_full_clean=False, new_data=None):
    """
    优化锁粒度：仅在读写Parquet时加锁，计算逻辑并行
    :param raw_path: 原始数据路径
    :param cleaned_path: 处理后数据保存路径
    :param lock: 文件锁对象（外部传入，确保单例）
    :param is_full_clean: 是否全量处理
    :param new_data: 单条新数据
    :return: 处理后的数据
    """
    try:
        required_fields = ["device_number", "create_time", "read_num"]

        # 全量处理（初始化时用）
        if is_full_clean:
            if not raw_path.exists():
                logger.error(f"原始数据不存在: {raw_path}")
                return None
            
            raw_df = pd.read_excel(raw_path)
            if not set(required_fields).issubset(raw_df.columns):
                logger.error(f"原始数据缺少字段: {required_fields}")
                return None
            
            # 格式转换（计算逻辑，无锁）
            cleaned_df = pd.DataFrame()
            cleaned_df["device_number"] = raw_df["device_number"].astype(str)
            cleaned_df["create_time"] = raw_df["create_time"].astype(str)
            cleaned_df["create_time_obj"] = pd.to_datetime(
                cleaned_df["create_time"], format="%Y%m%d", errors="coerce"
            )
            cleaned_df["unix_timestamp"] = cleaned_df["create_time_obj"].apply(
                lambda x: x.timestamp() if pd.notna(x) else np.nan
            )
            cleaned_df["read_num"] = pd.to_numeric(raw_df["read_num"], errors="coerce").fillna(0).round().astype(int)
            cleaned_df["daily_usage"] = 0.0
            cleaned_df["is_predicted"] = False

            # 写入时加锁
            with lock:
                cleaned_df.to_parquet(cleaned_path, index=False)
            logger.info(f"全量数据处理完成（{len(cleaned_df)}条）")
            return cleaned_df

        # 增量处理（新数据）
        else:
            if not new_data or not set(required_fields).issubset(new_data.keys()):
                logger.error("新数据缺少必要字段")
                return None

            # 格式转换和日期处理（计算逻辑，无锁）
            device_number = str(new_data["device_number"])
            create_time_str = str(new_data["create_time"])
            read_num = int(round(float(new_data["read_num"])))
            try:
                create_time_obj = datetime.strptime(create_time_str, "%Y%m%d")
            except ValueError:
                logger.error(f"日期格式错误: {create_time_str}")
                return None
            unix_timestamp = create_time_obj.timestamp()

            # 读取历史数据时加锁
            with lock:
                existing_df = pd.read_parquet(cleaned_path) if cleaned_path.exists() else pd.DataFrame()

            # 计算daily_usage（计算逻辑，无锁）
            daily_usage = 0.0
            if not existing_df.empty:
                prev_day_obj = create_time_obj - timedelta(days=1)
                prev_day_str = prev_day_obj.strftime("%Y%m%d")
                logger.info(f"当前日期: {create_time_str}, 前一天日期: {prev_day_str}")

                prev_day_data = existing_df[
                    (existing_df["device_number"] == device_number) & 
                    (existing_df["is_predicted"] == False) & 
                    (existing_df["create_time"] == prev_day_str)
                ]

                if not prev_day_data.empty:
                    last_read = prev_day_data.sort_values("unix_timestamp", ascending=False).iloc[0]["read_num"]
                    daily_usage = float(max(0.0, read_num - last_read))
                    logger.info(f"成功匹配前一天数据：前一日累加值={last_read}，当日累加值={read_num}，真实单日用量={daily_usage}")
                else:
                    device_history = existing_df[
                        (existing_df["device_number"] == device_number) & 
                        (existing_df["is_predicted"] == False)
                    ].sort_values("unix_timestamp")
                    if len(device_history) > 0:
                        last_read = device_history.iloc[-1]["read_num"]
                        daily_usage = float(max(0.0, read_num - last_read))
                        logger.warning(f"无前一天({prev_day_str})数据，使用历史最新数据计算：last_read={last_read}，daily_usage={daily_usage}")
                    else:
                        logger.info("无历史数据，daily_usage设为0")
            else:
                logger.info("无历史数据，daily_usage设为0")

            # 组装结果（计算逻辑，无锁）
            cleaned_single = {
                "device_number": device_number,
                "create_time": create_time_str,
                "create_time_obj": create_time_obj,
                "unix_timestamp": unix_timestamp,
                "read_num": read_num,
                "daily_usage": daily_usage,
                "is_predicted": False
            }

            # 写入时加锁
            with lock:
                if not existing_df.empty:
                    new_df = pd.DataFrame([cleaned_single])
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                    updated_df.to_parquet(cleaned_path, index=False)
                else:
                    pd.DataFrame([cleaned_single]).to_parquet(cleaned_path, index=False)

            logger.info(f"数据处理完成（设备{device_number}，日期{create_time_str}）")
            return cleaned_single

    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}", exc_info=True)
        return None