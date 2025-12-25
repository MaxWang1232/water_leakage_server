from flask import Flask, request, jsonify
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from mysql_water_leakage.mysql_config import get_mysql_connection, MySQLConfig

# 导入子模块（MySQL版本）
from mysql_water_leakage.mysql_cleaned_data_server import clean_water_data
from mysql_water_leakage.mysql_predict_new_data_server import WaterLeakagePredictor
from mysql_water_leakage.mysql_judger_server import WaterLeakageJudge 

app = Flask(__name__)

# 日志配置（与原代码完全一致）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_leakage_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """服务配置：保留Excel输入源，替换Parquet为MySQL配置"""
    RAW_DATA_PATH = Path("./test_data/test_water_leakage_data.xlsx")  # 原始Excel路径（全量初始化用）
    PREDICTION_OUTPUT_DIR = Path("./prediction_output")  # 预测结果目录
    # 数据库配置 - 实际使用时请修改为你的服务器信息
    DB_CONFIG = {
        "host": "10.65.49.205",
        "port": 13306,
        "user": "root",
        "password": "123456",
        "database": "water_leakage"
    }
    # 确保目录存在（与原代码一致）
    for path in [RAW_DATA_PATH.parent, PREDICTION_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


class WaterLeakageService:
    """水务漏损服务主类：功能与原代码完全一致，仅替换数据存储为MySQL"""
    def __init__(self):
        self.config = Config()
        self.judge = WaterLeakageJudge()  # 初始化判断模块
        self.predictor = None  # 预测器实例（后续初始化）
        self.mysql = MySQLConfig(**self.config.DB_CONFIG)  # MySQL连接实例
        self._init_service()  # 初始化服务

    def _init_service(self):
        """服务初始化：添加重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"\n===== 开始初始化水务漏损服务（MySQL版本）尝试 {attempt + 1}/{max_retries} =====")
                
                # 步骤1：若MySQL表为空，从Excel全量清洗数据
                conn = None
                cursor = None
                try:
                    conn = self.mysql.get_connection()
                    cursor = conn.cursor(buffered=True, dictionary=True)  # 使用缓冲游标
                    cursor.execute("SELECT COUNT(*) as count FROM water_data")
                    result = cursor.fetchone()  # 确保读取结果
                    count = result["count"]
                finally:
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
                
                if count == 0 and self.config.RAW_DATA_PATH.exists():
                    logger.info("MySQL表为空，从Excel执行全量清洗")
                    clean_result = clean_water_data(
                        raw_path=self.config.RAW_DATA_PATH,
                        db_config=self.mysql,
                        is_full_clean=True
                    )
                    if clean_result is None:
                        raise Exception("Excel全量清洗失败")
                    logger.info(f"全量清洗完成，生成{len(clean_result)}条数据")
                else:
                    logger.info(f"MySQL表已存在数据（{count}条），直接加载")
                
                # 步骤2：初始化预测器
                self.predictor = WaterLeakagePredictor(
                    mysql_config=self.mysql,
                    history_days=14,
                    output_dir=self.config.PREDICTION_OUTPUT_DIR
                )

                logger.info("===== 服务初始化完成，可接收请求 =====")
                break  # 成功则跳出重试循环
                
            except Exception as e:
                if attempt == max_retries - 1:
                    err_msg = f"服务初始化失败，已重试{max_retries}次: {str(e)}"
                    logger.error(err_msg, exc_info=True)
                    raise
                else:
                    logger.warning(f"服务初始化失败，{attempt + 1}次尝试: {str(e)}")
                    import time
                    time.sleep(5)  # 等待5秒后重试

    def process_request(self, input_data):
        """单设备请求处理：功能与原代码完全一致"""
        # 解析请求基础参数
        device_number = input_data["device_number"]
        input_date_str = input_data["create_time"]
        input_read_num = int(round(float(input_data["read_num"])))
        logger.info(f"\n===== 接收单设备请求：设备{device_number}（日期{input_date_str}） =====")
        
        try:
            # 1. 校验日期格式（与原代码一致）
            try:
                input_date_obj = datetime.strptime(input_date_str, "%Y%m%d")
            except ValueError:
                raise Exception(f"create_time格式错误，需为8位数字（如20251010），当前为{input_date_str}")

            # 2. 数据清洗（增量处理，调用MySQL版本清洗方法）
            cleaned_single = clean_water_data(
                raw_path=self.config.RAW_DATA_PATH,
                db_config=self.mysql,  # 传入MySQL配置
                is_full_clean=False,
                new_data=input_data
            )
            if cleaned_single is None:
                raise Exception("数据增量清洗失败（可能缺少必要字段或格式错误）")

            # 3. 调用预测器获取预测结果（MySQL版本预测方法）
            pred_data = self.predictor.predict_next_day(
                device_number=device_number,
                target_date=input_date_obj,
                current_read_num=input_read_num,
                real_daily_usage=cleaned_single["daily_usage"]
            )

            # 4. 调用判断模块生成报警信息（与原代码完全一致）
            real_data = {
                "device_number": device_number,
                "daily_usage": cleaned_single["daily_usage"],
                "create_time_obj": input_date_obj,
                "read_num": input_read_num
            }
            judge_result = self.judge.judge_single_device(
                pred_data=pred_data,
                history_data=self.predictor.get_device_history(device_number),
                real_data=real_data
            )

            # 5. 更新MySQL数据（调用MySQL版本更新方法）
            self.predictor.update_device_data(
                real_data=cleaned_single,
                pred_data=pred_data
            )

            # 6. 组装返回结果（与原代码字段完全一致）
            return {
                "alarm_info": judge_result["alarm_info"],
                "alarm_reason": judge_result["alarm_reason"],
                "daily_pred_usage": float(judge_result["daily_pred_usage"]),
                "daily_real_usage": float(round(cleaned_single["daily_usage"], 2)),
                "latest_pred_accumulated": float(judge_result["latest_pred_accumulated"]),
                "message": "请求处理成功"
            }

        except Exception as e:
            err_msg = f"单设备请求处理失败: {str(e)}"
            logger.error(err_msg, exc_info=True)
            raise

    def process_batch_request(self, batch_input):
        """批量设备请求处理：功能与原代码完全一致"""
        logger.info(f"\n===== 接收批量请求：共{len(batch_input)}个设备 =====")
        final_results = []
        valid_inputs = []

        # 步骤1：批量校验输入数据（与原代码完全一致）
        for idx, item in enumerate(batch_input):
            device_number = item.get("device_number", f"未知设备_{idx}")
            try:
                required_fields = ["device_number", "create_time", "read_num"]
                if not all(field in item for field in required_fields):
                    missing_fields = [f for f in required_fields if f not in item]
                    raise Exception(f"缺少必要字段：{','.join(missing_fields)}")
                
                create_time = item["create_time"]
                if not (isinstance(create_time, str) and len(create_time) == 8 and create_time.isdigit()):
                    raise Exception(f"create_time格式错误（需YYYYMMDD）：{create_time}")
                item["create_time_obj"] = datetime.strptime(create_time, "%Y%m%d")
                
                item["read_num"] = int(round(float(item["read_num"])))
                valid_inputs.append(item)
                logger.info(f"批量校验通过：设备{device_number}（日期{create_time}）")

            except Exception as e:
                err_msg = f"设备{device_number}校验失败：{str(e)}"
                logger.warning(err_msg)
                final_results.append({
                    "device_number": device_number,
                    "success": False,
                    "message": err_msg
                })

        if not valid_inputs:
            logger.warning("批量请求中无有效数据，直接返回")
            return final_results

        # 步骤2：批量清洗有效请求（调用MySQL版本清洗方法）
        cleaned_batch = []
        for item in valid_inputs:
            device_number = item["device_number"]
            try:
                cleaned = clean_water_data(
                    raw_path=self.config.RAW_DATA_PATH,
                    db_config=self.mysql,  # 传入MySQL配置
                    is_full_clean=False,
                    new_data=item
                )
                if not cleaned:
                    raise Exception("增量清洗后数据为空")
                cleaned_batch.append(cleaned)
                logger.info(f"设备{device_number}批量清洗完成")
            except Exception as e:
                err_msg = f"设备{device_number}批量清洗失败：{str(e)}"
                logger.error(err_msg, exc_info=True)
                final_results.append({
                    "device_number": device_number,
                    "success": False,
                    "message": err_msg
                })

        if not cleaned_batch:
            logger.warning("批量清洗后无有效数据，直接返回")
            return final_results

        # 步骤3：批量预测（调用MySQL版本批量预测方法）
        try:
            pred_input = [{
                "device_number": c["device_number"],
                "target_date": c["create_time_obj"],
                "current_read_num": c["read_num"],
                "real_daily_usage": c["daily_usage"]
            } for c in cleaned_batch]
            
            batch_preds = self.predictor.predict_batch(pred_input)
            pred_map = {p["device_number"]: p for p in batch_preds}
            logger.info(f"批量预测完成，共{len(batch_preds)}个设备")
        except Exception as e:
            err_msg = f"批量预测失败：{str(e)}"
            logger.error(err_msg, exc_info=True)
            for c in cleaned_batch:
                final_results.append({
                    "device_number": c["device_number"],
                    "success": False,
                    "message": f"{err_msg}（设备{c['device_number']}）"
                })
            return final_results

        # 步骤4：批量判断+更新数据+组装结果（与原代码一致）
        for cleaned in cleaned_batch:
            device_number = cleaned["device_number"]
            try:
                pred_data = pred_map.get(device_number)
                if not pred_data:
                    raise Exception("未找到对应预测结果（可能预测时遗漏）")
                
                real_data = {
                    "device_number": device_number,
                    "daily_usage": cleaned["daily_usage"],
                    "create_time_obj": cleaned["create_time_obj"],
                    "read_num": cleaned["read_num"]
                }
                judge_result = self.judge.judge_single_device(
                    pred_data=pred_data,
                    history_data=self.predictor.get_device_history(device_number),
                    real_data=real_data
                )
                
                # 更新MySQL数据
                self.predictor.update_device_data(
                    real_data=cleaned,
                    pred_data=pred_data
                )
                
                # 组装成功结果（字段与原代码完全一致）
                final_results.append({
                    "device_number": device_number,
                    "success": True,
                    "data": {
                        "alarm_info": judge_result["alarm_info"],
                        "alarm_reason": judge_result["alarm_reason"],
                        "daily_pred_usage": float(judge_result["daily_pred_usage"]),
                        "daily_real_usage": float(round(cleaned["daily_usage"], 2)),
                        "latest_pred_accumulated": float(judge_result["latest_pred_accumulated"]),
                        "message": "批量处理成功"
                    }
                })
                logger.info(f"设备{device_number}批量处理完成")

            except Exception as e:
                err_msg = f"设备{device_number}批量处理失败：{str(e)}"
                logger.error(err_msg, exc_info=True)
                final_results.append({
                    "device_number": device_number,
                    "success": False,
                    "message": err_msg
                })

        logger.info(f"===== 批量请求处理完成，共{len(final_results)}个结果 =====")
        return final_results


# 初始化服务实例（全局唯一）
service = WaterLeakageService()


# -------------------------- Flask接口定义（与原代码完全一致） --------------------------
@app.route('/WaterPredictor', methods=['POST'])
def water_predictor():
    """单设备请求接口：路由、请求响应格式与原代码完全一致"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "请求为空或格式错误（需JSON格式）"
            }), 400

        required_fields = ["device_number", "create_time", "read_num"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"缺少必要字段：{field}"
                }), 400

        result = service.process_request(data)
        return jsonify({
            "success": True,
            "data": result
        }), 200

    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"单设备接口异常：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "服务器内部错误（请查看日志）"
        }), 500


@app.route('/WaterPredictorBatch', methods=['POST'])
def water_predictor_batch():
    """批量设备请求接口：路由、请求响应格式与原代码完全一致"""
    try:
        batch_data = request.get_json()
        if not isinstance(batch_data, list):
            return jsonify({
                "success": False,
                "message": "请求格式错误（需为JSON列表，如[{...},{...}]）"
            }), 400

        result = service.process_batch_request(batch_data)
        return jsonify({
            "success": True,
            "total": len(result),
            "results": result
        }), 200

    except Exception as e:
        logger.error(f"批量接口异常：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "服务器内部错误（请查看日志）"
        }), 500


# -------------------------- 服务启动入口（与原代码一致） --------------------------
if __name__ == '__main__':
    logger.info("===== 启动Flask服务（MySQL版本，开发环境） =====")
    logger.info(f"服务地址：http://0.0.0.0:11027")
    logger.info(f"单设备接口：POST /WaterPredictor")
    logger.info(f"批量接口：POST /WaterPredictorBatch")
    app.run(host='0.0.0.0', port=11027, threaded=True, debug=False)