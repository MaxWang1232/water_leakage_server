import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_water_data(raw_path, db_config, is_full_clean=False, new_data=None):
    """
    数据处理：修复数据库连接和游标管理
    """
    try:
        required_fields = ["device_number", "create_time", "read_num"]
        mysql = db_config

        # 全量处理（初始化时用）
        if is_full_clean:
            if not raw_path.exists():
                logger.error(f"原始数据不存在: {raw_path}")
                return None
            
            # 读取Excel文件
            raw_df = pd.read_excel(raw_path)
            logger.info(f"读取到Excel数据，共{len(raw_df)}行，列名：{raw_df.columns.tolist()}")
            
            # 检查必要字段
            if not set(required_fields).issubset(raw_df.columns):
                logger.error(f"原始数据缺少字段: {required_fields}")
                return None
            
            # 数据清洗处理
            cleaned_df = pd.DataFrame()
            cleaned_df["device_number"] = raw_df["device_number"].astype(str)
            
            # 处理create_time
            cleaned_df["create_time_obj"] = pd.to_datetime(
                raw_df["create_time"], format="%Y%m%d", errors="coerce"
            )
            mask = cleaned_df["create_time_obj"].isna()
            cleaned_df.loc[mask, "create_time_obj"] = pd.to_datetime(
                raw_df.loc[mask, "create_time"], format="%Y-%m-%d", errors="coerce"
            )
            cleaned_df["create_time"] = cleaned_df["create_time_obj"].dt.strftime("%Y%m%d")
            
            cleaned_df["unix_timestamp"] = cleaned_df["create_time_obj"].apply(
                lambda x: x.timestamp() if pd.notna(x) else np.nan
            )
            
            cleaned_df["read_num"] = pd.to_numeric(raw_df["read_num"], errors="coerce").fillna(0).round().astype(int)
            cleaned_df["daily_usage"] = 0.0
            cleaned_df["is_predicted"] = False
            cleaned_df["daily_pred_usage"] = None
            cleaned_df["latest_pred_accumulated"] = None
            
            # 过滤无效日期数据
            valid_mask = cleaned_df["create_time_obj"].notna()
            invalid_count = len(cleaned_df) - valid_mask.sum()
            if invalid_count > 0:
                logger.warning(f"发现{invalid_count}条无效日期数据，将被跳过")
                cleaned_df = cleaned_df[valid_mask]
            
            if len(cleaned_df) == 0:
                logger.error("没有有效的数据可插入")
                return None
            
            # 分批插入数据库
            batch_size = 500  # 减小批次大小
            total_rows = len(cleaned_df)
            inserted_count = 0
            
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch_df = cleaned_df.iloc[start:end]
                
                conn = None
                cursor = None
                try:
                    conn = mysql.get_connection()
                    cursor = conn.cursor(buffered=True)  # 使用缓冲游标
                    
                    insert_query = """
                    INSERT INTO water_data 
                    (device_number, create_time, create_time_obj, unix_timestamp, 
                     read_num, daily_usage, is_predicted, daily_pred_usage, latest_pred_accumulated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    read_num = VALUES(read_num),
                    daily_usage = VALUES(daily_usage),
                    updated_at = CURRENT_TIMESTAMP
                    """
                    
                    batch_data = [
                        (
                            row.device_number, 
                            row.create_time, 
                            row.create_time_obj, 
                            row.unix_timestamp,
                            row.read_num, 
                            row.daily_usage, 
                            row.is_predicted, 
                            row.daily_pred_usage,
                            row.latest_pred_accumulated
                        )
                        for _, row in batch_df.iterrows()
                    ]
                    
                    cursor.executemany(insert_query, batch_data)
                    conn.commit()
                    
                    inserted_count += len(batch_data)
                    logger.info(f"插入批次 {start//batch_size + 1}/{(total_rows-1)//batch_size + 1}，行数 {start}-{end}")
                    
                except Exception as batch_error:
                    if conn:
                        conn.rollback()
                    logger.error(f"批次插入失败: {str(batch_error)}")
                    raise batch_error
                finally:
                    # 确保每次批次都正确关闭游标和连接
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
            
            logger.info(f"全量数据处理完成（{inserted_count}条有效数据）")
            return cleaned_df

        # 增量处理（新数据） - 这部分也需要修复
        else:
            if not new_data or not set(required_fields).issubset(new_data.keys()):
                logger.error("新数据缺少必要字段")
                return None

            # 处理device_number
            device_number = str(new_data["device_number"])

            # 处理create_time
            create_time_str = str(new_data["create_time"])
            create_time_obj = None
            try:
                create_time_obj = datetime.strptime(create_time_str, "%Y%m%d")
            except ValueError:
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"]:
                    try:
                        create_time_obj = datetime.strptime(create_time_str, fmt)
                        break
                    except ValueError:
                        continue
            
            if create_time_obj is None:
                logger.error(f"日期格式错误: {create_time_str}")
                return None
            
            standardized_create_time = create_time_obj.strftime("%Y%m%d")
            unix_timestamp = create_time_obj.timestamp()

            # 处理read_num
            try:
                read_num = int(round(float(new_data["read_num"])))
            except (ValueError, TypeError):
                logger.error(f"read_num格式错误: {new_data['read_num']}")
                return 0

            # 读取历史数据计算用水量
            conn = None
            cursor = None
            daily_usage = 0.0
            try:
                conn = mysql.get_connection()
                cursor = conn.cursor(buffered=True, dictionary=True)  # 使用缓冲游标
                
                prev_day_obj = create_time_obj - timedelta(days=1)
                prev_day_str = prev_day_obj.strftime("%Y%m%d")
                logger.info(f"当前日期: {standardized_create_time}, 前一天日期: {prev_day_str}")

                # 查找前一天数据
                cursor.execute("""
                SELECT read_num FROM water_data 
                WHERE device_number = %s 
                  AND is_predicted = FALSE 
                  AND create_time = %s
                ORDER BY unix_timestamp DESC
                LIMIT 1
                """, (device_number, prev_day_str))
                
                prev_day_data = cursor.fetchone()  # 确保读取结果

                if prev_day_data:
                    last_read = prev_day_data["read_num"]
                    daily_usage = float(max(0.0, read_num - last_read))
                    logger.info(f"成功匹配前一天数据：前一日累加值={last_read}，当日累加值={read_num}，真实单日用量={daily_usage}")
                else:
                    # 查找设备最新历史数据
                    cursor.execute("""
                    SELECT read_num FROM water_data 
                    WHERE device_number = %s 
                      AND is_predicted = FALSE
                    ORDER BY unix_timestamp DESC
                    LIMIT 1
                    """, (device_number,))
                    
                    last_record = cursor.fetchone()  # 确保读取结果
                    if last_record:
                        last_read = last_record["read_num"]
                        daily_usage = float(max(0.0, read_num - last_read))
                        logger.warning(f"无前一天({prev_day_str})数据，使用历史最新数据计算：last_read={last_read}，daily_usage={daily_usage}")
                    else:
                        logger.info("无历史数据，daily_usage设为0")

            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            # 组装结果
            cleaned_single = {
                "device_number": device_number,
                "create_time": standardized_create_time,
                "create_time_obj": create_time_obj,
                "unix_timestamp": unix_timestamp,
                "read_num": read_num,
                "daily_usage": daily_usage,
                "is_predicted": False,
                "daily_pred_usage": None,
                "latest_pred_accumulated": None
            }

            # 保存到数据库
            conn = None
            cursor = None
            try:
                conn = mysql.get_connection()
                cursor = conn.cursor(buffered=True)  # 使用缓冲游标
                
                insert_query = """
                INSERT INTO water_data 
                (device_number, create_time, create_time_obj, unix_timestamp, 
                 read_num, daily_usage, is_predicted, daily_pred_usage, latest_pred_accumulated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                read_num = VALUES(read_num),
                daily_usage = VALUES(daily_usage),
                updated_at = CURRENT_TIMESTAMP
                """
                
                cursor.execute(insert_query, (
                    device_number, standardized_create_time, create_time_obj, unix_timestamp,
                    read_num, daily_usage, False, None, None
                ))
                
                conn.commit()
                logger.info(f"数据处理完成（设备{device_number}，日期{standardized_create_time}）")
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            return cleaned_single

    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}", exc_info=True)
        return None