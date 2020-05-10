import pymysql
from datetime import datetime, date

DIM_AREA = 'dim_area'
DIM_PROD = 'dim_prod'
DIM_DATE = 'dim_date'
SALESYS = 'salesys'
SALESYSDW = 'salesys_dw'


class SimpleETL(object):

    def __init__(self):
        super().__init__()
        self.db_conn = None
        self.dw_conn = None

        # 获取数据库中的项目
        items = self._get_db_items()

        # 显示
        for item in items:
            print(item)
        print()


        # 获取key
        for item in items:
            order_id, prod, class_name, customer, area, created_time, total_cost = item
            # area key
            area_key = self._get_update_dim_key(DIM_AREA, area_name=area)
            # prod key
            prod_key = self._get_update_dim_key(DIM_PROD, prod_name=prod, class_name=class_name)
            # date key
            # created_time为datetime类型，修正为date类型
            created_time = created_time.date()
            created_key = self._get_update_dim_key(DIM_DATE, created_time=created_time)
            print('area, prod, created_time:', area_key, prod_key, created_key)

            # 构建事实表
            self._ins_fact_item(area_key, prod_key, created_key, total_cost)
        print('Fact update.')


    def _get_db_items(self):
        """获取数据库中的数据"""
        self.db_conn = self._connect(SALESYS)

        cursor = self.db_conn.cursor()
        # (订单号，商品，商品类别，客户名称，地区，创建日期，总价）
        sql = "select order_info.id, product.name, class.name, customer.name, area.name, order_table.created_time, total_cost" \
              " from class, product, order_info, order_table, customer, area" \
              " where class.id = product.class_id" \
              " and product.id = order_info.product_id" \
              " and order_info.order_id = order_table.id" \
              " and order_table.customer_id = customer.id" \
              " and customer.area_id = area.id"
        cursor.execute(sql)
        items = cursor.fetchall()

        self.db_conn.close()
        return items


    def _ins_fact_item(self, area_key, prod_key, created_key, total_cost):
        """插入事实表数据"""
        # 建立数据仓库连接
        self.dw_conn = self._connect(SALESYSDW)
        # 在数据仓库中查找数据，如果没有，则需要插入数据
        sql = "select * from fact_sale where area_key = %s" \
              " and prod_key = %s" \
              " and date_key = %s" \
              " and total_cost = %s" % (area_key, prod_key, created_key, total_cost)
        one = self._execute_dw(sql).fetchone()
        if not one:
            inssql = "insert into fact_sale(area_key, prod_key, date_key, total_cost)" \
                  " values(%s, %s, %s, %s )" % (area_key, prod_key, created_key, total_cost)
            self._execute_dw(inssql)
            self.dw_conn.commit()
        # 关闭数据仓库连接
        self.dw_conn.close()

    def _get_update_dim_key(self, dim_table=None, **kwargs):
        """
        获取维表key
        DIM_AREA --> area_name地区名称
        DIM_PROD --> prod_name商品名称， class_name商品类别
        DIM_DATE --> created_time创建时间
        :param dim_table:
        :param kwargs:
        :return:
        """
        try:
            # 打开数据仓库连接
            self.dw_conn = self._connect(SALESYSDW)

            key = None
            if dim_table == DIM_AREA:
                sql = "select * from dim_area where area = '%s'" % kwargs['area_name']
                ins_sql = "insert into dim_area(area) values('%s')" % kwargs['area_name']
            elif dim_table == DIM_PROD:
                sql = "select * from dim_prod where name = '%s' and class = '%s'" % (
                kwargs['prod_name'], kwargs['class_name'])
                ins_sql = "insert into dim_prod(name, class) values('%s', '%s')" % (
                kwargs['prod_name'], kwargs['class_name'])
            elif dim_table == DIM_DATE:
                sql = "select * from dim_date where date = '%s'" % kwargs['created_time']
                # 日期分解，日期为date类型
                year, month = kwargs['created_time'].year, kwargs['created_time'].month
                quard = (month - 1) // 3 + 1
                ins_sql = "insert into dim_date(date, year, quard, month) values('%s', '%s', '%s','%s')" \
                          % (kwargs['created_time'], year, quard, month)

            # 获取key
            cursor = self._execute_dw(sql)
            one = cursor.fetchone()
            if not one:  # 如果数据仓库中没有该数据，创建
                self._execute_dw(ins_sql)
                self.dw_conn.commit()
                key = self._execute_dw(sql).fetchone()[0]
            else:  # 如果数据仓库中有数据，返回key
                key = one[0]
        except Exception as e:
            print('Some error:', e)
        finally:
            # 关闭数据仓库连接
            self.dw_conn.close()
            pass

        return key

    def _execute_dw(self, sql):
        # self.dw_conn = self._connect(SALESYSDW)
        with self.dw_conn.cursor() as cursor:
            cursor.execute(sql)
        # self.dw_conn.close()
        return cursor

    def _connect(self, datebase_name=None):
        # 与数据库建立连接
        if datebase_name:
            conn = pymysql.connect(
                host='localhost',
                user='root',
                password='8088',
                database=datebase_name,
            )
            return conn


if __name__ == '__main__':
    SimpleETL()
    print('Done.')
