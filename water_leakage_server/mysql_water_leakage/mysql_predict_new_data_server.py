import logging
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from mysql_water_leakage.mysql_config import get_mysql_connection

logger = logging.getLogger(__name__)

class WaterLeakagePredictor:
    def __init__(self, mysql_config, history_days=14, output_dir=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        self.mysql = mysql_config  # 数据库配置
        self.history_days = history_days
        self.output_dir = output_dir or Path("./predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_data = self._load_cleaned_data()
        self.model = self._init_lstm_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _load_cleaned_data(self):
        """从MySQL加载历史数据"""
        try:
            conn = self.mysql.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("SELECT * FROM water_data ORDER BY unix_timestamp")
            data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if not data:
                return pd.DataFrame(columns=[
                    "device_number", "create_time", "create_time_obj", "unix_timestamp",
                    "read_num", "daily_usage", "is_predicted",
                    "daily_pred_usage", "latest_pred_accumulated"
                ])
            
            df = pd.DataFrame(data)
            df["device_number"] = df["device_number"].astype(str)
            df["create_time_obj"] = pd.to_datetime(df["create_time_obj"])
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=[
                "device_number", "create_time", "create_time_obj", "unix_timestamp",
                "read_num", "daily_usage", "is_predicted",
                "daily_pred_usage", "latest_pred_accumulated"
            ])

    def _init_lstm_model(self):
        """初始化LSTM模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(self.history_days, 1)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def _remove_outliers(self, data):
        """移除异常值"""
        if len(data) < 5:
            return data
        data_np = np.array(data)
        mean = np.mean(data_np)
        std = np.std(data_np)
        if std == 0:
            return data_np.tolist()
        z_scores = np.abs((data_np - mean) / std)
        return data_np[z_scores <= 3].tolist()

    def _fill_missing_dates(self, device_history, target_date):
        """填充缺失日期数据"""
        start_date = target_date - timedelta(days=self.history_days)
        date_range = pd.date_range(start=start_date, end=target_date - timedelta(days=1))
        full_dates = pd.DataFrame({"create_time_obj": date_range})

        history_df = device_history[["create_time_obj", "read_num", "daily_usage"]].copy()
        merged = pd.merge(full_dates, history_df, on="create_time_obj", how="left")

        real_usages = history_df["daily_usage"].dropna()
        mean_usage = real_usages.mean() if len(real_usages) > 0 else 0.3
        min_usage = max(0.1, real_usages.min()) if len(real_usages) > 0 else 0.1

        if pd.isna(merged.iloc[0]["read_num"]):
            first_valid = history_df["read_num"].dropna().min() if not history_df.empty else 0
            merged.iloc[0, merged.columns.get_loc("read_num")] = first_valid
            merged.iloc[0, merged.columns.get_loc("daily_usage")] = min_usage
        
        mask = merged["read_num"].isna()
        if mask.any():
            prev_read = merged["read_num"].shift(1)
            offset_low = max(0.1, mean_usage * 0.2) if mean_usage else 0.1
            offset_high = max(offset_low + 0.1, mean_usage * 0.5) if mean_usage else 0.2
            random_offsets = np.random.uniform(offset_low, offset_high, size=len(merged))
            merged.loc[mask, "read_num"] = prev_read[mask] + random_offsets[mask]
            merged.loc[mask, "daily_usage"] = random_offsets[mask]
            
        return merged

    def _count_prev_zero_days(self, device_history):
        """统计历史零用量天数"""
        if len(device_history) < 2:
            return 0
        real_history = device_history[device_history["is_predicted"] == False].sort_values("create_time_obj")
        zero_days = 0
        for i in range(1, len(real_history)):
            if real_history.iloc[i]["read_num"] == real_history.iloc[i-1]["read_num"]:
                zero_days += 1
        return zero_days

    def get_device_history(self, device_number):
        """获取设备历史数据"""
        conn = self.mysql.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
        SELECT * FROM water_data 
        WHERE device_number = %s 
        ORDER BY unix_timestamp
        """, (str(device_number),))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not data:
            return pd.DataFrame(columns=self.cleaned_data.columns)
            
        df = pd.DataFrame(data)
        df["create_time_obj"] = pd.to_datetime(df["create_time_obj"])
        return df.sort_values("unix_timestamp").copy()

    def predict_next_day(self, device_number, target_date, current_read_num, real_daily_usage):
        """预测次日用水量"""
        device_number = str(device_number)
        device_history = self.get_device_history(device_number)
        
        last_read = current_read_num - real_daily_usage
        last_read = max(0, last_read)
        
        # 准备历史数据
        filled_history = self._fill_missing_dates(device_history, target_date)
        daily_usages = filled_history["daily_usage"].values[-self.history_days:]

        # 处理异常值
        filtered_usages = self._remove_outliers(daily_usages)
        if len(filtered_usages) < self.history_days:
            fill_val = np.mean(filtered_usages) if len(filtered_usages) > 0 else random.uniform(0.1, 0.5)
            filtered_usages = np.pad(
                filtered_usages,
                (0, self.history_days - len(filtered_usages)),
                mode='constant',
                constant_values=fill_val
            )

        # LSTM预测
        scaled_data = self.scaler.fit_transform(np.array(filtered_usages).reshape(-1, 1))
        x_input = scaled_data.reshape(1, self.history_days, 1)
        pred_scaled = self.model.predict(x_input, verbose=0)[0][0]
        daily_pred_usage = self.scaler.inverse_transform([[pred_scaled]])[0][0]

        # 处理预测结果
        daily_pred_usage = round(daily_pred_usage, 2)
        daily_pred_usage = max(0.1, daily_pred_usage)
        latest_pred_accumulated = round(last_read + daily_pred_usage, 2)

        # 处理预测值为0的情况
        if latest_pred_accumulated == last_read:
            logger.info(f"设备{device_number}预测值为0，触发调整逻辑")
            zero_days = self._count_prev_zero_days(device_history)
            base = 1.0 / (zero_days if zero_days > 0 else 3)
            base = min(base, 0.6)
            random_offset = random.uniform(0.1, 0.4)
            adjusted_daily = base + random_offset

            adjusted_daily = max(adjusted_daily, 0.1)
            adjusted_daily = min(adjusted_daily, 0.8)
            adjusted_daily = round(adjusted_daily, 2)

            daily_pred_usage = float(adjusted_daily)
            latest_pred_accumulated = round(last_read + daily_pred_usage, 2)
            logger.info(f"调整后：单日用量={daily_pred_usage}，累加值={latest_pred_accumulated}")
        
        # 增加随机性
        random_add = random.uniform(0, 0.5)
        daily_pred_usage += random_add
        daily_pred_usage = round(daily_pred_usage, 2)
        latest_pred_accumulated = round(last_read + daily_pred_usage, 2)
        
        # 组装结果
        result = {
            "device_number": device_number,
            "predict_date": target_date.strftime("%Y%m%d"),
            "predict_date_obj": target_date,
            "daily_pred_usage": daily_pred_usage,
            "latest_pred_accumulated": latest_pred_accumulated
        }
        logger.info(f"设备{device_number}预测完成：单日用量={daily_pred_usage}，累加值={latest_pred_accumulated}")
        return result

    def predict_batch(self, batch_data):
        """批量预测"""
        results = []
        X = []
        device_info = []

        for data in batch_data:
            device_number = str(data["device_number"])
            target_date = data["target_date"]
            current_read_num = data["current_read_num"]
            real_daily_usage = data["real_daily_usage"]

            device_history = self.get_device_history(device_number)
            last_read = max(0, current_read_num - real_daily_usage)
            filled_history = self._fill_missing_dates(device_history, target_date)
            daily_usages = filled_history["daily_usage"].values[-self.history_days:]
            filtered_usages = self._remove_outliers(daily_usages)

            if len(filtered_usages) < self.history_days:
                fill_val = np.mean(filtered_usages) if filtered_usages else np.random.uniform(0.1, 0.5)
                filtered_usages = np.pad(
                    filtered_usages, (0, self.history_days - len(filtered_usages)),
                    mode="constant", constant_values=fill_val
                )

            X.append(filtered_usages)
            device_info.append({
                "device_number": device_number,
                "target_date": target_date,
                "last_read": last_read,
                "device_history": device_history
            })

        # 批量预测
        X_np = np.array(X)
        X_scaled = self.scaler.fit_transform(X_np.reshape(-1, 1)).reshape(-1, self.history_days, 1)
        pred_scaled = self.model.predict(X_scaled, verbose=0).flatten()

        # 处理结果
        for i in range(len(batch_data)):
            info = device_info[i]
            daily_pred_usage = self.scaler.inverse_transform([[pred_scaled[i]]])[0][0]
            daily_pred_usage = max(0.1, round(daily_pred_usage, 2))
            latest_pred_accumulated = round(info["last_read"] + daily_pred_usage, 2)

            if latest_pred_accumulated == info["last_read"]:
                zero_days = self._count_prev_zero_days(info["device_history"])
                base = min(1.0 / (zero_days or 3), 0.6)
                daily_pred_usage = round(max(0.1, min(0.8, base + np.random.uniform(0.1, 0.4))), 2)
                latest_pred_accumulated = round(info["last_read"] + daily_pred_usage, 2)

            daily_pred_usage = round(daily_pred_usage + np.random.uniform(0, 0.5), 2)
            latest_pred_accumulated = round(info["last_read"] + daily_pred_usage, 2)

            results.append({
                "device_number": info["device_number"],
                "predict_date": info["target_date"].strftime("%Y%m%d"),
                "predict_date_obj": info["target_date"],
                "daily_pred_usage": daily_pred_usage,
                "latest_pred_accumulated": latest_pred_accumulated
            })

        return results

    def update_device_data(self, real_data, pred_data):
        """更新设备数据到MySQL"""
        try:
            conn = self.mysql.get_connection()
            cursor = conn.cursor()
            
            update_query = """
            UPDATE water_data 
            SET daily_pred_usage = %s, 
                latest_pred_accumulated = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE device_number = %s AND create_time = %s
            """
            
            cursor.execute(update_query, (
                pred_data["daily_pred_usage"],
                pred_data["latest_pred_accumulated"],
                real_data["device_number"],
                real_data["create_time"]
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"设备{real_data['device_number']}数据已更新")

        except Exception as e:
            logger.error(f"更新设备数据失败：{str(e)}", exc_info=True)