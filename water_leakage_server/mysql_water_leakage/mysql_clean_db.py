from mysql_water_leakage.mysql_config import get_mysql_connection

def clean_database():
    mysql = get_mysql_connection()
    conn = mysql.get_connection()
    cursor = conn.cursor()
    
    try:
        # 直接清空表
        cursor.execute("TRUNCATE TABLE water_data")
        conn.commit()
        print("数据库已清空")
    except Exception as e:
        print(f"错误: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    clean_database()