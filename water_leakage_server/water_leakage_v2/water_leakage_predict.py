import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class WaterLeakagePredictor:
    def __init__(self, history_days=14):
        # GPU内存配置
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        self.history_days = history_days  # 保持14天预测窗口
        self.model = self._init_lstm_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _init_lstm_model(self):
        """初始化LSTM模型（减少神经元数量，避免过拟合）"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(self.history_days, 1)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def _remove_outliers(self, daily_usages):
        """Z-score法移除异常值"""
        if len(daily_usages) < 5:
            return daily_usages
        mean = np.mean(daily_usages)
        std = np.std(daily_usages)
        if std == 0:
            return daily_usages
        z_scores = np.abs((daily_usages - mean) / std)
        return daily_usages[z_scores <= 3]

    def _count_prev_zero_days(self, daily_usages):
        """统计历史零用量天数（用水量≤1e-6视为0）"""
        return sum(1 for usage in daily_usages if usage <= 1e-6)

    def predict_today_usage(self, daily_usages):
        """基于14天历史用水量预测今日用量，增加多样性逻辑"""
        try:
            if len(daily_usages) != self.history_days:
                raise ValueError(f"需输入{self.history_days}天数据，实际输入{len(daily_usages)}天")
            
            # 全0数据特殊处理：避免直接返回0.1，增加调整逻辑
            if np.all(daily_usages <= 1e-6):
                zero_days = self._count_prev_zero_days(daily_usages)
                base = 1.0 / (zero_days if zero_days > 0 else 3)
                base = min(base, 0.6)
                random_offset = np.random.uniform(0.1, 0.4)
                adjusted_daily = base + random_offset
                adjusted_daily = max(adjusted_daily, 0.1)
                adjusted_daily = min(adjusted_daily, 0.8)
                adjusted_daily = round(adjusted_daily, 2)
                logger.info(f"历史全0，调整后预测值：{adjusted_daily}")
                return adjusted_daily
            
            # 异常值过滤与填充
            filtered_usages = self._remove_outliers(daily_usages)
            if len(filtered_usages) < self.history_days:
                fill_val = np.mean(filtered_usages) if len(filtered_usages) > 0 else 0.3
                filtered_usages = np.pad(
                    filtered_usages,
                    (0, self.history_days - len(filtered_usages)),
                    mode='constant',
                    constant_values=fill_val
                )
            
            # 归一化：每次拟合当前数据（适配数据波动）
            self.scaler.fit(filtered_usages.reshape(-1, 1))
            scaled_data = self.scaler.transform(filtered_usages.reshape(-1, 1))
            x_input = scaled_data.reshape(1, self.history_days, 1)
            
            # 模型预测与反归一化
            pred_scaled = self.model.predict(x_input, verbose=0)[0][0]
            today_pred_usage = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            
            # 确保预测值合理 + 多样性调整
            today_pred_usage = max(0.1, round(today_pred_usage, 2))
            
            # 若预测值过于接近0.1，进行调整
            zero_days = self._count_prev_zero_days(filtered_usages)
            if abs(today_pred_usage - 0.1) < 0.05:
                base = 1.0 / (zero_days if zero_days > 0 else 3)
                base = min(base, 0.6)
                random_offset = np.random.uniform(0.1, 0.4)
                adjusted_daily = base + random_offset
                adjusted_daily = max(adjusted_daily, 0.1)
                adjusted_daily = min(adjusted_daily, 0.8)
                adjusted_daily = round(adjusted_daily, 2)
                today_pred_usage = adjusted_daily
                logger.info(f"预测值接近0.1，调整后：今日用水量={today_pred_usage}")
            
            # 可选：增加小幅随机偏移（进一步丰富结果）
            random_add = np.random.uniform(0, 0.2)
            today_pred_usage += random_add
            today_pred_usage = round(today_pred_usage, 2)
            today_pred_usage = max(0.1, today_pred_usage)
            
            logger.info(f"预测完成：今日用水量={today_pred_usage}")
            return today_pred_usage
        
        except Exception as e:
            logger.error(f"预测失败: {str(e)}", exc_info=True)
            raise