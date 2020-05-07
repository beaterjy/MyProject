from django.contrib import admin
from .models import Post, Category, Tag


class PostAdmin(admin.ModelAdmin):
    # Post后台类，继承ModelAdmin，通过register(Post, PostAdmin)关联Post
    # 相当于视图函数

    # 显示的列
    list_display = ['title', 'created_time', 'modified_time', 'category', 'author']
    # 显示的字段
    fields = ['title', 'body', 'excerpt', 'category', 'tags']

    def save_model(self, request, obj, form, change):
        """
        复写save_model，自动添加作者
        :param request: 保存按钮的请求，包含管理员对象user
        :param obj: 管理页面上填写保存的信息
        :param form:
        :param change:
        :return:
        """
        # 自动添加管理员作者
        obj.author = request.user
        super().save_model(request, obj, form, change)


admin.site.register(Post, PostAdmin)    # Post绑定后台类PostAdmin
admin.site.register(Category)
admin.site.register(Tag)


