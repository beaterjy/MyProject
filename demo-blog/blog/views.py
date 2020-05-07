from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.utils.text import slugify
from .models import Post, Category, Tag
import markdown
from markdown.extensions.toc import TocExtension


def index(request):
    post_list = Post.objects.all().order_by('-created_time')
    return render(request, 'blog/index.html', context={
        'post_list': post_list,
    })


def detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    md = markdown.Markdown(extensions=[
        'markdown.extensions.extra',  # 基础扩展
        'markdown.extensions.codehilite',  # 代码高亮扩展
        # 'markdown.extensions.fenced_code',
        # 'markdown.extensions.toc',  # 自动生成目录
        TocExtension(slugify=slugify)  # 代替toc，处理markdown无法处理的锚点问题(#_1, #_3)
    ])
    post.body = md.convert(post.body)  # markdown转换成html
    post.toc = md.toc  # 动态添加post的toc属性，也是转换成的html
    return render(request, 'blog/detail.html', context={'post': post, })


def archive(request, year, month):
    """归档视图函数"""
    post_list = Post.objects.filter(created_time__year=year,
                                    created_time__month=month
                                    ).order_by('-created_time')
    return render(request, 'blog/index.html', context={
        'post_list': post_list,
    })


def category(request, pk):
    """分类视图函数"""
    cate = get_object_or_404(Category, pk=pk)
    post_list = Post.objects.filter(category=cate).order_by('-created_time')
    return render(request, 'blog/index.html', context={
        'post_list': post_list
    })


def tag(request, pk):
    """标签视图函数"""
    t = get_object_or_404(Tag, pk=pk)
    post_list = Post.objects.filter(tags=t).order_by('-created_time')
    return render(request, 'blog/index.html', context={
        'post_list': post_list
    })
