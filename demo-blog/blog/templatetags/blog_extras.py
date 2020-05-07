from django import template
from ..models import Post, Category, Tag

register = template.Library()

# 最新文章模板标签
# {% show_recent_posts 5 %}其中5为参数num的值
@register.inclusion_tag('blog/inclusions/_recent_posts.html', takes_context=True)   # context为父模板的上下文
def show_recent_posts(context, num=5):
    return {
        'recent_post_list': Post.objects.all().order_by('-created_time')[:num],
    }


# 归档模板标签
@register.inclusion_tag('blog/inclusions/_archives.html', takes_context=True)   # context为父模板的上下文
def show_archives(context):
    return {
        # order为'DESC'代表descending, 'ASC'代表ascending
        # month为精度
        # created_time为指定的域field
        'date_list': Post.objects.dates('created_time', 'month', order='DESC')
    }


# 分类模板标签
@register.inclusion_tag('blog/inclusions/_categories.html', takes_context=True)   # context为父模板的上下文
def show_categories(context):
    return {
        'category_list': Category.objects.all(),
    }

# 标签云模板标签
@register.inclusion_tag('blog/inclusions/_tags.html', takes_context=True)   # context为父模板的上下文
def show_tags(context):
    return {
        'tag_list': Tag.objects.all(),
    }

