import logging
import numpy as np

logger = logging.getLogger(__name__)

class WaterLeakageJudge:
    """水务漏损检测判断类（基于真实用量、预测用量及21天历史数据）"""
    def __init__(self):
        logger.info("初始化漏损判断模块（支持21天历史数据）")

    def judge(self, real_today_usage, pred_today_usage, history_daily_usages):
        """
        基于真实用量、预测用量和21天历史用量判断漏损
        :param real_today_usage: 今日真实用水量
        :param pred_today_usage: 预测今日用水量
        :param history_daily_usages: 21天历史日用水量（np.array，最近1天在前）
        :return: 包含报警信息的字典
        """
        result = {
            "alarm_info": "无报警",
            "alarm_reason": [],
        }
        logger.info(
            f"开始判断：真实今日用量={real_today_usage}, "
            f"预测今日用量={pred_today_usage}, 21天历史用水量={history_daily_usages}"
        )

        # 前置判断逻辑：当日真实值需大于前一日真实值2吨以上才进入后续判断
        # 前一日真实值为历史数据的第一个元素（最近1天在前）
        if len(history_daily_usages) < 1:
            logger.info("无历史前一日数据，不满足前置判断条件，直接返回无报警")
            return result
        
        prev_day_usage = history_daily_usages[0]
        if real_today_usage <= (prev_day_usage + 1):
            logger.info(
                f"今日真实用量({real_today_usage})未超过前一日用量({prev_day_usage})+1吨，"
                "不满足前置判断条件，直接返回无报警"
            )
            return result
        
        logger.info(
            f"今日真实用量({real_today_usage})超过前一日用量({prev_day_usage})+2吨，"
            "满足前置判断条件，进入后续漏损检测"
        )

        # 计算历史统计特征（保留原有14天特征，新增21天特征）
        # 最近14天特征（用于兼容原有逻辑）
        recent_14 = history_daily_usages[:14] if len(history_daily_usages)>=14 else history_daily_usages
        mean_14 = np.mean(recent_14) + 1e-6  # 避免除零
        max_3 = np.max(history_daily_usages[:3]) if len(history_daily_usages)>=3 else 0
        
        # 21天特征（新增）
        mean_21 = np.mean(history_daily_usages) + 1e-6  # 21天均值
        max_7 = np.max(history_daily_usages[:7]) if len(history_daily_usages)>=7 else 0  # 最近7天最大值
        prev_7_zero = all(usage <= 1e-6 for usage in history_daily_usages[:7]) if len(history_daily_usages)>=7 else False
        prev_14_zero = all(usage <= 1e-6 for usage in history_daily_usages[:14]) if len(history_daily_usages)>=14 else False

        # 报警条件判断（按优先级排序，新增21天相关条件）
        if real_today_usage > 20 * pred_today_usage:
            result["alarm_info"] = "重度报警"
            result["alarm_reason"].append(
                f"今日真实用量({real_today_usage}) > 20倍预测用量)"
            )
        elif real_today_usage > 20:
            result["alarm_info"] = "重度报警"
            result["alarm_reason"].append(
                f"今日真实用量({real_today_usage}) > 20吨阈值"
            )
        elif real_today_usage > 30 * mean_21:  
            result["alarm_info"] = "中度报警"
            result["alarm_reason"].append(
                f"今日真实用量({real_today_usage}) > 21天均值的30倍"
            )
        elif real_today_usage > 30 * mean_14:
            result["alarm_info"] = "中度报警"
            result["alarm_reason"].append(
                f"今日真实用量({real_today_usage}) > 最近14天均值的30倍"
            )
        elif len(history_daily_usages)>=7 and real_today_usage > 4.5 * max_7:  
            result["alarm_info"] = "中度报警"
            result["alarm_reason"].append(
                f"今日预测用量({real_today_usage}) > 最近7天最大值的4倍"
            )
        elif len(history_daily_usages)>=3 and real_today_usage > 4.5 * max_3:
            result["alarm_info"] = "中度报警"
            result["alarm_reason"].append(
                f"今日预测用量({real_today_usage}) > 最近3天最大值的4倍"
            )
        elif prev_14_zero and real_today_usage > 1e-6:  
            result["alarm_info"] = "轻微提醒"
            result["alarm_reason"].append(
                f"最近14天全为0，今日真实用量({real_today_usage})恢复使用"
            )
        elif prev_7_zero and real_today_usage > 1e-6:
            result["alarm_info"] = "轻微提醒"
            result["alarm_reason"].append(
                f"最近7天全为0，今日真实用量({real_today_usage})恢复使用"
            )
        else:
            pass

        logger.info(f"判断结果：{result['alarm_info']}，原因：{'; '.join(result['alarm_reason'])}")
        return result