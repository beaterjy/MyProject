from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.urls import reverse
from django.utils.html import strip_tags    # 去除html标签
import markdown


class Category(models.Model):
    """
    分类
    """
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Tag(models.Model):
    """
    标签
    """
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name


class Post(models.Model):
    """
    文章
    """
    # 文章标题，小文本
    title = models.CharField(max_length=70)

    # 文章正文，大文本，用TextField
    body = models.TextField()

    # 文章创建时间和最后一次修改时间
    created_time = models.DateTimeField(default=timezone.now)   # default没填写时的默认值，可以为常量，可以为可调用对象(timezone.now)
                                                                # timezone.now()会在创建时直接产生，可能与保存到数据库中时的值产生时差
    modified_time = models.DateTimeField()

    # 文章摘要，可以没有文章摘要，blank=True
    excerpt = models.CharField(max_length=200, blank=True)

    # 分类与标签
    # 分类，一对多，级联删除
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    # 标签，多对多，可以为空
    tags = models.ManyToManyField(Tag, blank=True)

    # 作者，一对多，使用内置User类，级联删除
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    # Meta内部类，通过指定属性的值为Post对象添加默认特性
    class Meta:
        ordering = ['-created_time']    # 默认按照创建时间倒序

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        """复写save方法，自动更新修改时间"""
        self.modified_time = timezone.now()

        # 自动生成摘要 -->
        # 1.先markdown转换成html；
        # 2.去掉html标签；
        # 3.提取前N个字符作为摘要。
        md = markdown.Markdown(extensions=[     # 不需要用到toc文章目录扩展
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
        ])
        self.excerpt = strip_tags(md.convert(self.body))[:54]   # N为54

        super().save(*args, **kwargs)

    def get_absolute_url(self):
        """返回当前post的detail的url地址"""
        return reverse('blog:detail', kwargs={'pk': self.pk})

