from flask import Flask, request, jsonify
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from filelock import FileLock

# 导入子模块
from cleaned_data_server import clean_water_data
from predict_new_data_server import WaterLeakagePredictor
from judger_server import WaterLeakageJudge

app = Flask(__name__)

# 日志配置（确保日志正常输出）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_leakage_service.log'),  # 日志写入文件
        logging.StreamHandler()  # 日志打印到控制台
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """服务配置：保留Excel输入源和Parquet存储路径"""
    RAW_DATA_PATH = Path("./test_data/test_water_leakage_data.xlsx")  # 原始Excel路径
    CLEANED_DATA_PATH = Path("./cleaned_data/cleaned_water_leakage_data.parquet")  # Parquet存储路径
    PREDICTION_OUTPUT_DIR = Path("./prediction_output")  # 预测结果目录
    # 确保所有目录存在（避免路径不存在报错）
    for path in [RAW_DATA_PATH.parent, CLEANED_DATA_PATH.parent, PREDICTION_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


class WaterLeakageService:
    """水务漏损服务主类：优化锁粒度，支持多线程并发"""
    def __init__(self):
        self.config = Config()
        self.judge = WaterLeakageJudge()  # 初始化判断模块
        self.predictor = None  # 预测器实例（后续初始化）
        # 单例文件锁：仅在读写Parquet时加锁，全局唯一避免重复创建
        self.cleaned_data_lock = FileLock(str(self.config.CLEANED_DATA_PATH) + ".lock")
        self._init_service()  # 初始化服务

    def _init_service(self):
        """服务初始化：全量清洗Excel（首次启动）、初始化预测器"""
        try:
            logger.info("\n===== 开始初始化水务漏损服务 =====")
            
            # 步骤1：若Parquet文件不存在，从Excel全量清洗数据
            if not self.config.CLEANED_DATA_PATH.exists():
                logger.info("未找到Parquet数据文件，从Excel执行全量清洗")
                clean_result = clean_water_data(
                    raw_path=self.config.RAW_DATA_PATH,
                    cleaned_path=self.config.CLEANED_DATA_PATH,
                    lock=self.cleaned_data_lock,  # 传入全局锁
                    is_full_clean=True
                )
                if clean_result is None:
                    raise Exception("Excel全量清洗失败（可能Excel文件缺失或格式错误）")
                logger.info(f"全量清洗完成，生成{len(clean_result)}条数据")
            else:
                logger.info(f"已找到Parquet数据文件，直接加载（路径：{self.config.CLEANED_DATA_PATH}）")
            
            # 步骤2：初始化预测器（加载Parquet历史数据）
            self.predictor = WaterLeakagePredictor(
                cleaned_data_path=self.config.CLEANED_DATA_PATH,
                history_days=14,  # 预测依赖的历史天数（保持原逻辑）
                output_dir=self.config.PREDICTION_OUTPUT_DIR
            )

            logger.info("===== 服务初始化完成，可接收请求 =====")

        except Exception as e:
            err_msg = f"服务初始化失败: {str(e)}"
            logger.error(err_msg, exc_info=True)  # 打印详细错误栈
            raise  # 抛出异常，终止服务启动

    def process_request(self, input_data):
        """
        单设备请求处理：清洗→预测→判断→更新
        输入字段：device_number/create_time/read_num
        输出字段：alarm_info/alarm_reason/daily_pred_usage/daily_real_usage/latest_pred_accumulated
        """
        # 解析请求基础参数
        device_number = input_data["device_number"]
        input_date_str = input_data["create_time"]
        input_read_num = int(round(float(input_data["read_num"])))
        logger.info(f"\n===== 接收单设备请求：设备{device_number}（日期{input_date_str}） =====")
        
        try:
            # 1. 校验日期格式（确保为YYYYMMDD）
            try:
                input_date_obj = datetime.strptime(input_date_str, "%Y%m%d")
            except ValueError:
                raise Exception(f"create_time格式错误，需为8位数字（如20251010），当前为{input_date_str}")

            # 2. 数据清洗（增量处理，仅读写Parquet时加锁）
            cleaned_single = clean_water_data(
                raw_path=self.config.RAW_DATA_PATH,  # 增量处理时仅用于参数兼容，实际不读取
                cleaned_path=self.config.CLEANED_DATA_PATH,
                lock=self.cleaned_data_lock,  # 传入全局锁，仅在读写时加锁
                is_full_clean=False,
                new_data=input_data
            )
            if cleaned_single is None:
                raise Exception("数据增量清洗失败（可能缺少必要字段或格式错误）")

            # 3. 调用预测器获取预测结果
            pred_data = self.predictor.predict_next_day(
                device_number=device_number,
                target_date=input_date_obj,
                current_read_num=input_read_num,
                real_daily_usage=cleaned_single["daily_usage"]  # 清洗后得到的真实单日用量
            )

            # 4. 调用判断模块生成报警信息
            real_data = {
                "device_number": device_number,
                "daily_usage": cleaned_single["daily_usage"],  # 真实单日用量
                "create_time_obj": input_date_obj,
                "read_num": input_read_num  # 真实累加值
            }
            judge_result = self.judge.judge_single_device(
                pred_data=pred_data,  # 预测结果
                history_data=self.predictor.get_device_history(device_number),  # 设备历史数据
                real_data=real_data  # 真实数据
            )

            # 5. 更新Parquet文件（将真实数据+预测结果写入，仅加锁读写）
            self.predictor.update_device_data(
                real_data=cleaned_single,
                pred_data=pred_data,
                lock=self.cleaned_data_lock,  # 传入全局锁
                cleaned_path=self.config.CLEANED_DATA_PATH
            )

            # 6. 组装返回结果（严格匹配需求字段）
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
            raise  # 抛出异常，让接口返回500错误

    def process_batch_request(self, batch_input):
        """批量设备请求处理：支持多设备同时提交，逻辑与单设备一致"""
        logger.info(f"\n===== 接收批量请求：共{len(batch_input)}个设备 =====")
        final_results = []  # 存储最终批量结果
        valid_inputs = []   # 存储校验通过的请求

        # 步骤1：批量校验输入数据（过滤无效请求）
        for idx, item in enumerate(batch_input):
            device_number = item.get("device_number", f"未知设备_{idx}")
            try:
                # 校验必要字段
                required_fields = ["device_number", "create_time", "read_num"]
                if not all(field in item for field in required_fields):
                    missing_fields = [f for f in required_fields if f not in item]
                    raise Exception(f"缺少必要字段：{','.join(missing_fields)}")
                
                # 校验日期格式
                create_time = item["create_time"]
                if not (isinstance(create_time, str) and len(create_time) == 8 and create_time.isdigit()):
                    raise Exception(f"create_time格式错误（需YYYYMMDD）：{create_time}")
                item["create_time_obj"] = datetime.strptime(create_time, "%Y%m%d")
                
                # 校验read_num格式（转为整数）
                item["read_num"] = int(round(float(item["read_num"])))
                
                # 校验通过，加入有效列表
                valid_inputs.append(item)
                logger.info(f"批量校验通过：设备{device_number}（日期{create_time}）")

            except Exception as e:
                # 无效请求，记录错误信息
                err_msg = f"设备{device_number}校验失败：{str(e)}"
                logger.warning(err_msg)
                final_results.append({
                    "device_number": device_number,
                    "success": False,
                    "message": err_msg
                })

        # 若没有有效请求，直接返回结果
        if not valid_inputs:
            logger.warning("批量请求中无有效数据，直接返回")
            return final_results

        # 步骤2：批量清洗有效请求
        cleaned_batch = []  # 存储清洗后的有效数据
        for item in valid_inputs:
            device_number = item["device_number"]
            try:
                cleaned = clean_water_data(
                    raw_path=self.config.RAW_DATA_PATH,
                    cleaned_path=self.config.CLEANED_DATA_PATH,
                    lock=self.cleaned_data_lock,
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

        # 若清洗后无有效数据，直接返回
        if not cleaned_batch:
            logger.warning("批量清洗后无有效数据，直接返回")
            return final_results

        # 步骤3：批量预测（调用预测器的批量方法）
        try:
            # 组装预测输入参数
            pred_input = [{
                "device_number": c["device_number"],
                "target_date": c["create_time_obj"],
                "current_read_num": c["read_num"],
                "real_daily_usage": c["daily_usage"]
            } for c in cleaned_batch]
            
            # 调用批量预测方法
            batch_preds = self.predictor.predict_batch(pred_input)
            # 转为字典，方便按设备号匹配
            pred_map = {p["device_number"]: p for p in batch_preds}
            logger.info(f"批量预测完成，共{len(batch_preds)}个设备")
        except Exception as e:
            err_msg = f"批量预测失败：{str(e)}"
            logger.error(err_msg, exc_info=True)
            # 为所有清洗后的设备添加预测失败信息
            for c in cleaned_batch:
                final_results.append({
                    "device_number": c["device_number"],
                    "success": False,
                    "message": f"{err_msg}（设备{device_number}）"
                })
            return final_results

        # 步骤4：批量判断+更新数据+组装结果
        for cleaned in cleaned_batch:
            device_number = cleaned["device_number"]
            try:
                # 匹配当前设备的预测结果
                pred_data = pred_map.get(device_number)
                if not pred_data:
                    raise Exception("未找到对应预测结果（可能预测时遗漏）")
                
                # 批量判断报警信息
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
                
                # 批量更新Parquet数据
                self.predictor.update_device_data(
                    real_data=cleaned,
                    pred_data=pred_data,
                    lock=self.cleaned_data_lock,
                    cleaned_path=self.config.CLEANED_DATA_PATH
                )
                
                # 组装当前设备的成功结果
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

        # 步骤5：返回批量结果
        logger.info(f"===== 批量请求处理完成，共{len(final_results)}个结果 =====")
        return final_results


# 初始化服务实例（全局唯一，避免重复创建）
service = WaterLeakageService()


# -------------------------- Flask接口定义 --------------------------
@app.route('/WaterPredictor', methods=['POST'])
def water_predictor():
    """单设备请求接口：接收JSON请求，返回指定字段"""
    try:
        # 1. 解析JSON请求
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "请求为空或格式错误（需JSON格式）"
            }), 400  # 400：请求参数错误

        # 2. 校验必要字段
        required_fields = ["device_number", "create_time", "read_num"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"缺少必要字段：{field}"
                }), 400

        # 3. 处理请求并返回结果
        result = service.process_request(data)
        return jsonify({
            "success": True,
            "data": result
        }), 200  # 200：请求成功

    except ValueError as ve:
        # 已知的参数错误（如日期格式）
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        # 未知的服务器错误
        logger.error(f"单设备接口异常：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "服务器内部错误（请查看日志）"
        }), 500  # 500：服务器错误


@app.route('/WaterPredictorBatch', methods=['POST'])
def water_predictor_batch():
    """批量设备请求接口：接收JSON列表，返回批量结果"""
    try:
        # 1. 解析JSON请求（需为列表格式）
        batch_data = request.get_json()
        if not isinstance(batch_data, list):
            return jsonify({
                "success": False,
                "message": "请求格式错误（需为JSON列表，如[{...},{...}]）"
            }), 400

        # 2. 处理批量请求
        result = service.process_batch_request(batch_data)
        return jsonify({
            "success": True,
            "total": len(result),  # 总结果数
            "results": result      # 详细结果列表
        }), 200

    except Exception as e:
        # 服务器错误
        logger.error(f"批量接口异常：{str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": "服务器内部错误（请查看日志）"
        }), 500


# -------------------------- 服务启动入口 --------------------------
if __name__ == '__main__':
    # 开发环境：Flask内置服务器（开启多线程）
    logger.info("===== 启动Flask服务（开发环境） =====")
    logger.info(f"服务地址：http://0.0.0.0:11027")
    logger.info(f"单设备接口：POST /WaterPredictor")
    logger.info(f"批量接口：POST /WaterPredictorBatch")
    # threaded=True：开启多线程，支持并发处理请求
    app.run(host='0.0.0.0', port=11027, threaded=True, debug=False)  # debug=False：生产环境禁用调试模式
