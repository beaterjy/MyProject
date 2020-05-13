"""
数据库工具包
"""
import pymysql

DIM_AREA = 'dim_area'
DIM_PROD = 'dim_prod'
DIM_DATE = 'dim_date'
SALESYS = 'salesys'
SALESYSDW = 'salesys_dw'

class DB(object):

    def __init__(self, database_name=SALESYSDW):
        super().__init__()
        self.database_name = database_name

    def _connect(self, database_name=None):
        # 与数据库建立连接
        if database_name:
            conn = pymysql.connect(
                host='localhost',
                user='root',
                password='8088',
                database=database_name,
            )
            return conn

    # 主要方法：运行sql语句，返回结果
    def get_items(self, sql):

        # 连接数据库
        self.dw_conn = self._connect(self.database_name)

        with self.dw_conn.cursor() as cursor:
            cursor.execute(sql)
            items = cursor.fetchall()

        # 关闭数据库
        self.dw_conn.close()
        return items

