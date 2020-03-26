# 使用说明

仅有两个python文件

simpleRTT.py

test_client.py

## 用于本机测试

simpleRTT.py中默认使用的端口是PORT=12321

默认使用本机回送地址127.0.0.1

## 用于远程测试

通过修改simpleRTT.py文件中
（均为外网地址）

Server属性为远程ip地址

Client属性为本机ip地址

## 运行代码
先python3 simpleRTT.py作为服务器

然后python3 test_client.py作为客户端
