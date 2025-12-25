from flask import Flask, request, jsonify
import logging
import numpy as np
from datetime import datetime, timedelta
from water_leakage_predict import WaterLeakagePredictor
from water_leakage_judge import WaterLeakageJudge
from collections import OrderedDict

app = Flask(__name__)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化预测器（14天窗口）和判断器
predictor = WaterLeakagePredictor(history_days=14)
judger = WaterLeakageJudge()

def is_continuous_dates(dates):
    """检查日期列表是否连续（按降序排列，相邻日期相差1天）"""
    for i in range(len(dates) - 1):
        # 计算相邻日期差（应为1天）
        day_diff = (dates[i] - dates[i+1]).days
        if day_diff != 1:
            return False
    return True

def calculate_history_daily_usages(read_nums):
    """
    计算过去21天的日用水量（read_nums[0]是最近一天，read_nums[20]是最老）
    :param read_nums: 21天累计读数列表（按日期降序排列）
    :return: 21天日用水量列表（np.array）
    """
    daily_usages = []
    for i in range(len(read_nums)):
        if i == len(read_nums) - 1:  # 最老一天无更早数据，用量设为0
            daily = max(0, read_nums[i] - read_nums[i])
        else:  # 第i天用量 = 第i天累计 - 第i+1天累计
            daily = max(0, read_nums[i] - read_nums[i+1])
        daily_usages.append(round(daily, 2))
    return np.array(daily_usages)

@app.route('/WaterPredictor', methods=['POST'])
def water_predict():
    """
    单设备预测接口：基于连续21天历史数据预测最新日期用水量并判断漏损
    输入格式：{
        "device_number": "xxx",
        "data": [
            {"create_time": "20251103", "read_num": "2"},  # 最新日期
            {"create_time": "20251102", "read_num": "2"},  # 前一天
            ...,  # 共21条记录（最新+过去20天）
            {"create_time": "20251014", "read_num": "2"}
        ]
    }
    输出格式：{
        "device_number": "xxx",
        "create_time": "最新日期",
        "read_num": 最新日期累计读数,
        "data": {
            "alarm_info": "...",
            "alarm_reason": [...],
            "message": "...",
            "predict_cumulative": ...,
            "predict_daily_usage": ...,
            "real_daily_usage": ...
        },
        "success": true
    }
    """
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({
                "success": False,
                "message": "请求为空或格式错误（需JSON）"
            }), 400

        # 校验顶层必要字段
        required_top = ["device_number", "data"]
        missing_top = [f for f in required_top if f not in input_data]
        if missing_top:
            return jsonify({
                "success": False,
                "message": f"缺少必要顶层字段：{','.join(missing_top)}"
            }), 400

        # 校验data数组
        data_list = input_data["data"]
        if not isinstance(data_list, list) or len(data_list) != 21:
            return jsonify({
                "success": False,
                "message": f"data应为包含21条记录的数组，实际{len(data_list)}条"
            }), 400

        # 解析data中的日期和读数，转换为datetime和float
        parsed_data = []
        for idx, item in enumerate(data_list):
            # 校验单条记录字段
            if not isinstance(item, dict) or "create_time" not in item or "read_num" not in item:
                return jsonify({
                    "success": False,
                    "message": f"data第{idx+1}条记录缺少create_time或read_num"
                }), 400
            # 转换日期
            try:
                create_time = datetime.strptime(item["create_time"], "%Y%m%d")
            except ValueError:
                return jsonify({
                    "success": False,
                    "message": f"data第{idx+1}条记录日期格式错误（应为YYYYMMDD）"
                }), 400
            # 转换读数
            try:
                read_num = float(item["read_num"])
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "message": f"data第{idx+1}条记录read_num格式错误（需数字）"
                }), 400
            parsed_data.append({
                "create_time": create_time,
                "read_num": round(read_num, 2)
            })

        # 按日期降序排序（最新日期在前）
        parsed_data.sort(key=lambda x: x["create_time"], reverse=True)
        dates = [item["create_time"] for item in parsed_data]
        read_nums = [item["read_num"] for item in parsed_data]

        # 检查日期连续性
        if not is_continuous_dates(dates):
            return jsonify({
                "success": False,
                "message": "预测序列日非连续，再次输入"
            }), 400

        # 提取关键信息
        device_number = input_data["device_number"]
        latest_date = dates[0].strftime("%Y%m%d")  # 最新日期（预测目标日期）
        latest_read_num = read_nums[0]  # 最新日期累计读数
        history_read_nums = read_nums[1:22]  # 过去21天累计读数（read_num_1到read_num_21）

        # 计算21天历史日用水量
        history_daily_usages_21 = calculate_history_daily_usages(history_read_nums)
        logger.info(f"设备{device_number}：21天历史日用水量={history_daily_usages_21}")

        # 取最近14天用于预测
        history_daily_usages_14 = history_daily_usages_21[:14]
        if len(history_daily_usages_14) < 14:
            raise ValueError(f"历史数据不足14天（实际{len(history_daily_usages_14)}天）")

        # 预测今日用水量
        pred_today_usage = predictor.predict_today_usage(history_daily_usages_14)

        # 计算今日真实用水量（最新日期累计 - 昨天累计）
        real_today_usage = max(0, round(latest_read_num - history_read_nums[0], 2))
        logger.info(f"今日真实用水量={real_today_usage}，预测今日用水量={pred_today_usage}")

        # 计算预测累计值（昨天累计 + 预测今日用量）
        predict_cumulative = round(history_read_nums[0] + pred_today_usage, 2)

        # 漏损判断（传入21天历史数据）
        judge_result = judger.judge(real_today_usage, pred_today_usage, history_daily_usages_21)

        # 组装返回结果
# 组装返回结果时用 OrderedDict
        return jsonify(OrderedDict([
            ("device_number", device_number),
            ("create_time", latest_date),
            ("read_num", latest_read_num),
            ("data", OrderedDict([
                ("alarm_info", judge_result["alarm_info"]),
                ("alarm_reason", judge_result["alarm_reason"]),
                ("message", "预测判断完成"),
                ("latest_pred_accumulated", predict_cumulative),
                ("daily_pred_usage", pred_today_usage),
                ("daily_real_usage", real_today_usage)
            ])),
            ("success", True)
        ]))

    except Exception as e:
        logger.error(f"接口处理失败: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "服务器内部错误"
        }), 500

if __name__ == '__main__':
    logger.info("启动水务预测服务（支持21天连续日期数据）")
    logger.info("接口地址：POST /WaterPredictor")
    app.run(host='0.0.0.0', port=11027, debug=False)