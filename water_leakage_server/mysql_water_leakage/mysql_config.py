import mysql.connector
from mysql.connector import pooling
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class MySQLConfig:
    """MySQL数据库配置与连接池管理"""
    def __init__(self, host="10.65.49.205", port=13306, user="root", password="123456", database="water_leakage"):
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "autocommit": False,
            "auth_plugin": "mysql_native_password",
            "use_pure": True,
            "connect_timeout": 30,
            # 添加连接参数解决 Unread result found
            "buffered": True,  # 使用缓冲游标
            "consume_results": True,  # 自动消费结果
        }
        self.pool = self._create_connection_pool()
        self._create_tables()

    def _create_connection_pool(self):
        """创建数据库连接池"""
        try:
            pool_config = self.config.copy()
            # 移除连接池不支持的参数
            for key in ['buffered', 'consume_results']:
                if key in pool_config:
                    del pool_config[key]
            
            return pooling.MySQLConnectionPool(
                pool_name="water_pool",
                pool_size=16,  # 连接池大小 > 最大并发数（如12），预留缓冲
                pool_reset_session=True,  # 复用连接前重置会话（避免残留状态）
                **pool_config
            )
        except Exception as e:
            logger.error(f"创建数据库连接池失败: {str(e)}", exc_info=True)
            raise

    def get_connection(self):
        """从连接池获取连接（优化版：超时控制+异常细分+竞争缓解）"""
        max_retries = 3
        timeout_total = 5  # 总超时时间（秒）
        start_time = time.time()
    
        for attempt in range(max_retries):
            try:
                # 从连接池获取连接
                conn = self.pool.get_connection()
            
                # 验证连接有效性
                if not conn.is_connected():
                    conn.reconnect(attempts=1, delay=1)  # 主动重连
            
                # 测试连接可用性（执行心跳查询）
                with conn.cursor(buffered=True) as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchall()  # 确保消费所有结果
                return conn
        
            except mysql.connector.PoolError as e:
                # 连接池耗尽场景：短暂等待后重试，缓解竞争
                logger.warning(f"连接池资源耗尽，第 {attempt + 1}/{max_retries} 次尝试: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # 等待0.5秒后重试
                # 检查是否超时
                if time.time() - start_time > timeout_total:
                    break
        
            except Exception as e:
                # 其他异常（如网络波动、认证失败等）：直接重试
                logger.warning(f"获取连接异常（非连接池原因），第 {attempt + 1}/{max_retries} 次尝试: {str(e)}")
    
        # 最后一次尝试：直接创建新连接（绕过连接池，应急用）
        try:
            logger.warning("连接池获取失败，尝试直接创建新连接")
            conn = mysql.connector.connect(** self.config)
            # 验证新连接
            with conn.cursor(buffered=True) as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()
            return conn
        except Exception as final_e:
            logger.error(f"最终获取数据库连接失败: {str(final_e)}")
            raise

    # def get_connection(self):
    #     """从连接池获取连接（修复版本）"""
    #     max_retries = 3  # 减少重试次数
    #     for attempt in range(max_retries):
    #         try:
    #             conn = self.pool.get_connection()
    #             if conn.is_connected():
    #                 # 测试连接是否有效
    #                 cursor = conn.cursor(buffered=True)  # 使用缓冲游标
    #                 cursor.execute("SELECT 1")
    #                 cursor.fetchall()  # 确保读取所有结果
    #                 cursor.close()
    #                 return conn
    #             else:
    #                 conn.reconnect(attempts=1, delay=1)
    #                 return conn
    #         except Exception as e:
    #             logger.warning(f"获取数据库连接失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
    #             if attempt == max_retries - 1:
    #                 # 最后一次重试，创建新连接
    #                 try:
    #                     conn = mysql.connector.connect(**self.config)
    #                     return conn
    #                 except Exception as final_e:
    #                     logger.error(f"最终获取数据库连接失败: {str(final_e)}")
    #                     raise

    def _create_tables(self):
        """创建必要的数据表"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(buffered=True)  # 使用缓冲游标
            
            # 创建水数据主表
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS water_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                device_number VARCHAR(50) NOT NULL,
                create_time VARCHAR(8) NOT NULL,
                create_time_obj DATETIME NOT NULL,
                unix_timestamp DOUBLE NOT NULL,
                read_num INT NOT NULL,
                daily_usage DOUBLE NOT NULL,
                is_predicted BOOLEAN NOT NULL DEFAULT FALSE,
                daily_pred_usage DOUBLE NULL,
                latest_pred_accumulated DOUBLE NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_device_date (device_number, create_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            
            conn.commit()
            logger.info("数据库表初始化完成")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"创建数据库表失败: {str(e)}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# 默认配置
def get_mysql_connection():
    """获取数据库连接实例"""
    return MySQLConfig(
        host="10.65.49.205",
        port=13306,
        user="root",
        password="123456",
        database="water_leakage"
    )