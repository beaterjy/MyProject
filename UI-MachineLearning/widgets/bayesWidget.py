import tkinter as tk
from tkinter import scrolledtext, filedialog
from preprocessing import NLPs as _nlps
import docx
import pickle



class BayesWidget(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.eval_mess = ['优', '良', '中', '及格', '不及格']
        self._core = _nlps(self.eval_mess)
        self._val = tk.IntVar()
        self.label_stat = self._core.get_label_stat()
        self._initWidget(master)

    # 初始化bayes界面
    def _initWidget(self, master):
        # title frame (top)
        # content frame (bottom)
        self._frm_top = tk.Frame(master)
        self._frm_bottom = tk.Frame(master)

        '''top frame --> title'''
        tk.Label(self._frm_top, text='Naive Bayes 文本分类系统', font=('Consolas', 14),  width=30, height=5).pack()

        '''bottom frame --> content'''
        # left frame and right frame
        self._frm_left = tk.Frame(master=self._frm_bottom, padx=10)
        self._frm_right = tk.Frame(master=self._frm_bottom, padx=10)

        '''left frame --> 文档展示'''
        tk.Label(self._frm_left, text='当前文档', width=10, height=2).pack()

        # 滑动窗
        # self.textBox = tk.Text(self._frm_left, height=20, width=50)  # TODO: 滑动窗口
        self.textBox = scrolledtext.ScrolledText(self._frm_left, font=('Consolas', 14), height=10, width=40, relief=tk.GROOVE)

        # 按钮
        self._frm_btns = tk.Frame(master=self._frm_left)
        btn_func = [self._load_file, None, None,
                    self._save_model, self._load_model, None,
                    None, None, None,  ]
        mess = ['打开文档', '保存数据集', '装入数据集', '保存模型', '装入模型', '备用按钮', '备用按钮','备用按钮','备用按钮']
        self.btns = [tk.Button(self._frm_btns, text=m, pady=3, width=10, command=btn_func[idx])
                     for idx, m in enumerate(mess)]
        rows, cols = 3, 3
        for r in range(rows):
            for c in range(cols):
                self.btns[cols * r + c].grid(row=r, column=c, padx=20, pady=5)

        self.textBox.pack()
        self._frm_btns.pack()

        '''right frame --> 操作选项和结果'''
        # result frame, prob frame, choose frame, add frame, bayes frame
        self._frm_result = tk.Frame(self._frm_right, )
        self._frm_prob = tk.Frame(self._frm_right, borderwidth=2, relief='ridge')
        self._frm_choose = tk.Frame(self._frm_right, borderwidth=2, relief='ridge')
        self._frm_add = tk.Frame(self._frm_right)
        self._frm_bayes = tk.Frame(self._frm_right)

        # 显示分类结果
        tk.Label(self._frm_result, text='分类结果', ).pack(side='left', pady=30)
        self.result = tk.Entry(self._frm_result, justify='center')
        self.result.pack(side='right')

        # 显示分类可能性
        self.probs = [tk.Entry(self._frm_prob, width=10, justify='center') for _ in range(len(self.eval_mess))]
        for i in range(len(self.eval_mess)):
            self.probs[i].grid(row=0, column=i, padx=10, pady=5)
        # 分类标签
        for i in range(len(self.eval_mess)):
            tk.Label(self._frm_prob, text=self.eval_mess[i]).grid(row=1, column=i)

        # 当前各种标签数量
        self.labelCounts = [tk.Entry(self._frm_choose, width=10, justify='center') for _ in range(len(self.eval_mess))]
        for i in range(len(self.eval_mess)):
            self.labelCounts[i].grid(row=0, column=i, padx=10, pady=5)
        self._update_stat(component='LabelCount')

        # 分类单选按钮
        for i in range(len(self.eval_mess)):
            tk.Radiobutton(self._frm_choose, text=self.eval_mess[i],
                           value=i, variable=self._val).grid(row=1, column=i)


        # 添加样本按钮
        self.btn_add = tk.Button(self._frm_add, text='添加训练样本', command=self._add_sample).pack(pady=10)

        # bayes学习，分类按钮
        self.btn_learn = tk.Button(self._frm_bayes, text='贝叶斯学习', command=self._bayes_study).pack(side='left', padx=20, pady=10)
        self.btn_text = tk.Button(self._frm_bayes, text='贝叶斯分类', command=self._bayes_classify).pack(side='right', padx=20, pady=10)

        # frame pack
        self._frm_result.pack()
        self._frm_prob.pack(pady=30)
        self._frm_choose.pack()
        self._frm_add.pack()
        self._frm_bayes.pack()

        self._frm_left.pack(side='left')
        self._frm_right.pack(side='right')

        self._frm_top.pack()
        self._frm_bottom.pack()


    def _load_file(self):
        """装入文件"""
        filename = filedialog.askopenfilename(title='打开文件', initialdir='../files/',
                                          filetype=[('docx文件', '*.docx')])
        if len(filename) == 0:
            return None

        doc = docx.Document(filename)
        content = ''
        for  para in doc.paragraphs:
            content = content + '\n' + para.text

        # 显示文章
        self.textBox.delete(1.0, tk.END)
        self.textBox.insert(1.0, content)


    def _add_sample(self):
        """添加训练样本"""
        # 获取文本
        text = self.textBox.get(1.0, tk.END)

        # 保存数据
        self._core.add_data(text, self._val.get())

        # 标签数量更新
        self.label_stat[self._val.get()] += 1
        self._update_stat(component='LabelCount')

        # 清空文本
        self.textBox.delete(1.0, tk.END)
        print('Add sample Done.')

    def _update_stat(self, component=None):
        """
        更新组件数据
        :param components: 标签的统计 -- LabelCount
                            分类结果概率 -- Prob
                            分类结果 -- Result
        :param vals:
        :return:
        """
        # TODO: 补充内容
        if component == 'LabelCount':
            for i in range(len(self.labelCounts)):
                self.labelCounts[i].delete(0, tk.END)
                self.labelCounts[i].insert(0, self.label_stat[i])
        elif component == 'Prob':
            pass
        elif component == 'Result':
            pass

    def _bayes_study(self):
        """贝叶斯学习"""
        self._core.train()
        print('Bayes study Done.')

    def _bayes_classify(self):
        """贝叶斯分类"""
        # 获取文档
        text = self.textBox.get(1.0, tk.END)

        # 预处理 最终 编码成向量
        X_test = self._core.doc2vec(text).reshape((1, -1))

        # 分类
        pred = self._core.predict(X_test)

        # 显示结果
        self.result.delete(0, tk.END)
        self.result.insert(0, self.eval_mess[pred[0]])
        print('Bayes classify Done.')


    def _save_model(self):
        """保存模型"""
        filename = filedialog.asksaveasfilename(title='保存模型', initialdir='../files/model/',
                                                filetype=[('pk文件', '*.pk')])
        if len(filename) == 0:
            return

        if not filename.endswith('.pk'):
            filename = filename + '.pk'

        with open(filename, 'wb') as f:
            pickle.dump(self._core, f)

        print('Save model Done.')


    def _load_model(self):
        """加载模型"""
        filename = filedialog.askopenfilename(title='打开模型', initialdir='../files/model/',
                                              filetypes=[('pk文件', '*.pk')])
        if len(filename) == 0:
            return

        with open(filename, 'rb') as f:
            self._core = pickle.load(f)

        # 更新统计数据
        self.label_stat = self._core.get_label_stat()
        self._update_stat(component='LabelCount')

        print('Load model Done.')

if __name__ == '__main__':
    window = tk.Tk()
    root = BayesWidget().pack()
    window.mainloop()


