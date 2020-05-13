import tkinter as tk
from tkinter import ttk
import db_utils as _db

SQLs = {
    'dim_time': lambda x: 'select %s from dim_date'
                          ' group by %s'
                          ' order by %s' % (x, x, x),
    'dim_area': lambda x: 'select %s from dim_area'
                          ' group by %s'
                          ' order by %s' % (x, x, x),
    'dim_prod': lambda x: 'select %s from dim_prod'
                          ' group by %s'
                          ' order by %s' % (x, x, x),
    'cost': lambda gb, cond1, cond1_val, cond2, cond2_val: "select %s, sum(total_cost) from"
                                       " fact_sale natural join dim_area natural join dim_prod natural join dim_date"
                                       " where %s = '%s'"
                                       " and %s = '%s'"
                                       " group by %s"
                                       " order by %s" % (gb, cond1, cond1_val, cond2, cond2_val, gb, gb),
}

attr = {
    '日期': 'date',
    '年': 'year',
    '月': 'month',
    '季度': 'quard',
    '省份': 'prov',
    '市': 'area',
    '商品名': 'name',
    '类别': 'class',
}


class MainWidget(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.dw = _db.DB(database_name=_db.SALESYSDW)  # 创建于数据仓库的连接
        self._initWidget(master=master)

    def _initWidget(self, master):
        # dim frame, button frame, table frame
        frm_dim = tk.Frame(master)
        frm_btn = tk.Frame(master)
        self.frm_table = tk.Frame(master)

        # ============dim frame=============
        # 显示时间维，商品维，地区维标签
        tk.Label(frm_dim, text='时间维').grid(row=0, column=0)
        tk.Label(frm_dim, text='地区维').grid(row=0, column=1)
        tk.Label(frm_dim, text='商品维').grid(row=0, column=2)

        # 显示各个维的combobox
        self.time_val, self.prod_val, self.area_val = tk.StringVar(), tk.StringVar(), tk.StringVar()
        time_box = ttk.Combobox(frm_dim, width=12, textvariable=self.time_val)
        prod_box = ttk.Combobox(frm_dim, width=12, textvariable=self.prod_val)
        area_box = ttk.Combobox(frm_dim, width=12, textvariable=self.area_val)

        # 显示各个维的内容
        time_box['values'] = ('年', '季度', '月', )
        prod_box['values'] = ('商品名')
        area_box['values'] = ('省份', '市')
        # 默认内容
        time_box.current(0)
        prod_box.current(0)
        area_box.current(0)
        time_box.grid(row=1, column=0, padx=10, pady=5)
        area_box.grid(row=1, column=1, padx=10, pady=5)
        prod_box.grid(row=1, column=2, padx=10, pady=5)

        # ============button frame===========
        # 显示按钮
        btn = tk.Button(frm_btn, text='GO', width=20, height=1, command=self._update_table)

        # ============table frame===========

        # 显示表格
        columns = ('col1', 'col2')
        self.table = ttk.Treeview(self.frm_table, height=18, show='headings', columns=columns)
        self.table.column('col1', width=100, anchor='center')
        self.table.column('col2', width=100, anchor='center')

        # 显示表头
        self.table.heading('col1', text='姓名')
        self.table.heading('col2', text='IP地址')

        btn.pack(pady=10)
        self.table.pack()
        frm_dim.pack(padx=10, pady=5)
        frm_btn.pack(padx=10, pady=5)
        self.frm_table.pack(padx=10, pady=5)

    def _update_table(self):
        """更新表格"""
        # 获取combobox数据
        aTime = attr[self.time_val.get()]
        aArea = attr[self.area_val.get()]
        aProd = attr[self.prod_val.get()]

        # 清空 table frame 中内容
        for widget in self.frm_table.winfo_children():
            widget.destroy()

        # 最外层是商品维
        prod_items = self.dw.get_items(SQLs['dim_prod'](aProd))
        print('商品维:', prod_items)
        for pidx,  pitem in enumerate(prod_items):
            p = pitem[0]   # 商品维具体值
            # -------------sub title---------------
            tk.Label(self.frm_table, text=p).grid(row=0, column=pidx)

            # -------------sub table---------------
            # 构建columns
            area_items = self.dw.get_items(SQLs['dim_area'](aArea))
            columns = [x[0] for x in area_items]
            columns = sorted(list(set(columns)))
            if pidx == 0:       # 如果是第一个显示的列，需要添加时间维属性
                columns = [''] + columns    # 前端留一个空列放每一行信息
            print('columns:', columns)

            # 生成表格
            self.table = ttk.Treeview(self.frm_table, height=18, show='headings', columns=columns)

            # 构建table头部
            for c in columns:
                self.table.column(c, width=100, anchor='center')
                self.table.heading(c, text=c)

            # 构建table主体
            for r, t_item in enumerate(self.dw.get_items(SQLs['dim_time'](aTime))):
                val = ['' for i in range(len(columns))]
                t = t_item[0]   # 获取时间
                if pidx == 0:   # 如果是第一个显示的列，才需要添加时间维属性
                    val[0] = str(t)  # 第一项是每行区别信息

                sql = SQLs['cost'](gb=aArea, cond1=aTime, cond1_val=str(t), cond2=aProd, cond2_val=str(p))
                items = self.dw.get_items(sql)
                for item in items:
                    idx = columns.index(item[0])  # item第一项是索引
                    val[idx] = item[1]  # item第二项是值

                # 插入val
                self.table.insert('', r, values=val)

            # 表格显示
            self.table.grid(row=1, column=pidx)


if __name__ == '__main__':
    window = tk.Tk()
    MainWidget(window).pack()
    window.mainloop()
