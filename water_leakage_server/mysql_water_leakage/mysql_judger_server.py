import logging
import numpy as np
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

class WaterLeakageJudge:
    """水务漏损检测判断类"""
    def __init__(self):
        logger.info("===== 初始化判断模块 =====")

    def _is_prev_7_zero(self, real_history, target_date):
        """判断目标日期前7天是否全为0"""
        prev_data = real_history[real_history["create_time_obj"] < target_date].copy()
        if len(prev_data) < 7:
            logger.debug(f"历史数据不足7天（共{len(prev_data)}条），无法判断前7天全0")
            return False
        
        prev_7_days = prev_data.sort_values("create_time_obj").tail(7)
        all_zero = (prev_7_days["daily_usage"] <= 1e-6).all()
        
        if all_zero:
            logger.debug(f"目标日期{target_date.strftime('%Y%m%d')}的前7天全为0")
        else:
            logger.debug(f"目标日期{target_date.strftime('%Y%m%d')}的前7天存在非0用量")
        return all_zero

    def judge_single_device(self, pred_data, history_data, real_data):
        """单设备报警判断逻辑"""
        result = {
            "alarm_info": "无报警",
            "alarm_reason": [],
            "latest_pred_accumulated": pred_data["latest_pred_accumulated"],
            "daily_pred_usage": pred_data["daily_pred_usage"]
        }

        real_daily = real_data["daily_usage"]
        daily_pred = pred_data["daily_pred_usage"]
        real_history = history_data[history_data["is_predicted"] == False]
        target_date = real_data["create_time_obj"]
        device_number = real_data["device_number"]

        logger.info(f"\n===== 开始判断设备{device_number}（日期{target_date.strftime('%Y%m%d')}） =====")
        logger.info(f"判断参数：真实单日用量={real_daily}, 预测单日用量={daily_pred}, 历史数据量={len(real_history)}条")

        # 条件1：真实用量>20倍预测值 → 重度报警
        if real_daily > 20 * daily_pred:
            result["alarm_info"] = "重度报警"
            result["alarm_reason"].append(
                f"当日真实用量({real_daily}) > 20倍预测值({daily_pred})"
            )
        
        # 条件2：真实用量>20吨 → 重度报警
        elif real_daily > 20:
            result["alarm_info"] = "重度报警"
            result["alarm_reason"].append(
                f"单日用量({real_daily}) > 20吨阈值"
            )
        
        # 条件3：真实用量>14天均值*30 → 中度报警
        elif len(real_history) >= 14:
            past_14_daily = real_history["daily_usage"].tail(14).values
            mean_14 = float(np.mean(past_14_daily) + (1/14))
            if real_daily > 30 * mean_14:
                result["alarm_info"] = "中度报警"
                result["alarm_reason"].append(
                    f"当日用量({real_daily}) > 14天均值({mean_14:.2f})的30倍"
                )
        
        # 条件4：真实用量>前3天最大值*1.1 → 中度报警
        elif len(real_history) >= 3:
            past_3_daily = real_history["daily_usage"].tail(3).values
            max_3 = float(np.max(past_3_daily))
            if real_daily > 1.1 * max_3:
                result["alarm_info"] = "中度报警"
                result["alarm_reason"].append(
                    f"当日用量({real_daily}) > 前3天最大值({max_3:.2f})的1.1倍"
                )
        
        # 条件5：前7天全0且当日有用水 → 轻微提醒
        elif self._is_prev_7_zero(real_history, target_date) and real_daily > 0:
            result["alarm_info"] = "轻微提醒"
            result["alarm_reason"].append(
                f"前7天全为0，当日用量({real_daily})恢复使用"
            )
        
        # 条件6：所有条件不满足 → 无报警
        else:
            result["alarm_reason"].append(
                f"当日用量({real_daily})在正常范围内（≤20倍预测值{daily_pred}）"
            )

        logger.info(f"判断结果：{result['alarm_info']}，原因：{'; '.join(result['alarm_reason'])}")
        return result