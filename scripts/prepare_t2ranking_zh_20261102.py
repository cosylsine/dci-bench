#!/usr/bin/env python3
"""Prepare a high precision Chinese software slice of THUIR/T2Ranking.

The script is deliberately an offline preparer.  It reads only a complete
local T2Ranking snapshot, streams ``collection.tsv`` once, and publishes a
new DCI retrieval release after the source grades and retention decisions have
been written to a separate evidence directory.  It never treats a similarity
score as a gold label: only source grades 2 and 3 become binary qrels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import SamplingShortageError, sample_query_splits  # noqa: E402
from dci_bench.data.pipeline.core import load_bundle, publish_bundle, verify_release  # noqa: E402


DATASET = "THUIR/T2Ranking"
REVISION = "2a369a430a70979223f1b9a41b1919774d46b432"
LICENSE = "apache-2.0"
COLLECTION_LFS_SHA256 = "07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212"
COLLECTION_BYTES = 3_659_243_528
HF_BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
SOURCE_URLS = {
    "dataset_card": f"https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md",
    "paper": "https://arxiv.org/abs/2304.03679",
    "collection": f"{HF_BASE}/data/collection.tsv",
    "queries_dev": f"{HF_BASE}/data/queries.dev.tsv",
    "qrels_dev": f"{HF_BASE}/data/qrels.dev.tsv",
    "queries_train": f"{HF_BASE}/data/queries.train.tsv",
    "qrels_train": f"{HF_BASE}/data/qrels.train.tsv",
}
POLICY = {
    "max_positive_per_query": 200,
    "drop_near_duplicates": False,
}
FILTER_VERSION = "software-zh-high-precision-v5"
SCHEMA_VERSION = "dci-data-bundle-v1"
PROVENANCE_SOURCE_ID = "gapfill_t2ranking_software_zh"

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[\s\.,;:!?，。；：！？、（）()【】\[\]{}<>《》“”‘’'\"`~·|/\\+*=#@%^&_-]+")
NUM_RE = re.compile(r"\d+(?:[.]\d+)*")

# These are intentionally narrower than generic "computer" vocabulary.  A
# body must contain at least one technical anchor and enough independent
# Chinese prose to avoid retaining a code-only or consumer page.
TECHNICAL_TERMS: tuple[tuple[str, str], ...] = (
    ("language_or_runtime", r"(?<![A-Za-z0-9])(?:Python|Java(?:EE|SE|语言)?|JavaScript|C\+\+|C#|C语言|Go语言|Rust|PHP|Ruby|Kotlin|Swift|MATLAB|R语言|\.NET|Node(?:\.js)?|HTML|CSS|SQL)(?![A-Za-z0-9])"),
    ("software_stack", r"(?<![A-Za-z0-9])(?:MySQL|PostgreSQL|MongoDB|Redis|Linux|Unix|CentOS|Ubuntu|Git|Docker|Kubernetes|Nginx|Apache|Maven|Gradle|Spring(?:Boot|MVC)?|Django|Flask|React|Vue|Qt|Windows|Android|iOS)(?![A-Za-z0-9])|Android Studio|Android SDK|iOS SDK|Windows Server"),
    ("development", r"软件开发|软件工程|程序设计|编程开发|源代码|源码|编译|调试代码|调试程序|开发环境|开发工具|开发者|数据库|数据结构|命令行|Shell脚本|API接口|(?<![A-Za-z0-9])(?:API|SDK|IDE)(?![A-Za-z0-9])|前端|后端|全栈|版本控制|开源软件|包管理|Web开发|应用开发|接口调用|服务端|客户端|进程|端口|日志|权限管理|安装包"),
    ("systems_and_network", r"操作系统|内核|网络协议|(?<![A-Za-z0-9])(?:HTTP|HTTPS|TCP(?:/IP)?|UDP|DNS)(?![A-Za-z0-9])|代理服务器|云计算|虚拟机|容器化|模型部署|软件配置|配置文件|服务器|网络|域名|IP地址|(?<![A-Za-z0-9])IP(?![A-Za-z0-9])|端口|服务部署"),
)
TECHNICAL_PATTERNS = tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in TECHNICAL_TERMS)
CONSUMER_RE = re.compile(
    r"游戏|网游|手游|页游|棋牌游戏|外挂|破解版|破解补丁|激活码|序列号|电影|电视剧|音乐|歌曲|小说|直播|娱乐|壁纸|铃声|动漫",
    re.IGNORECASE,
)
HARDWARE_RE = re.compile(r"HDMI|AV接口|DB9|M2接口|插头|剃须刀|信号定义|显示器|主板|硬盘|磁盘|显卡|SCSI|硬件|内存|插槽|千兆|网卡|洗衣机|空调|电梯|热水器|遥控器|流水灯|PLC|伺服驱动器", re.IGNORECASE)
QUERY_EXCLUSION_RE = re.compile(
    r"院校|学院|大学|学校|专业|学籍|户籍|行政区划|行政区|地区代码|城市代码|区号|邮政编码|机构代码|组织机构代码|纳税人识别号|银行|中银|银行卡|swift\s*code|swift代码|swift码|BIC|IBAN|"
    r"股票|证券|股市|基金|期货|外汇|彩票|教师资格|驾照|压力容器|医学|药品|美食|装修|汽车配件|发票|土地使用权|"
    r"编程猫|培训机构|月薪|工资|就业|考证|课程|初学|入门|造句|翻译|背框架|政策框架|"
    r"洗衣机|空调|电梯|热水器|遥控器|手机补电|手机代码|摩托罗拉|三星note|华为手机|监控|录像机|摄像头|无信号|"
    r"PS4|iTunes|抢购|茅台|交通攻略|traffic rider|网络用语|网络语言|网络词|网络上是什么意思|网络科技公司|网络客服|网络女主播|网络审查|网络安全问题|网络利大于弊|网络电视|网络.*(?:是什么意思|用语|语言|词)|个人所得税|奇异果|支付宝客户端|百度网盘|美团网络异常|航空公司|国防|九九乘法表|两只老虎|九阳神功|DSD|DOP|我的世界|MC服务器|DNF|魔兽|吃鸡|王者荣耀|贪吃蛇|网络充值|充值.*IP|苹果|华为v?\d|移动设备|尚未绑定|央视少儿|小米|开发者(?:模式|选项)|按键精灵|网卡|电脑.*(?:端口|硬件|设备)|学习(?:Java|Python|编程)|好书|重要单词|Docker.*学多久|CAD.*安装包|域名代码|IP核|CAN通讯|老王vp|保养|指甲|人体|症状|心电图|qt间期|角色.*IP|IP设计|查别人.*IP|公司排名|揪痧|潮流计算|供电|Windows\s*Store",
    re.IGNORECASE,
)
QUERY_MOBILE_RE = re.compile(r"Android|安卓|iOS|iPhone|鸿蒙|苹果系统", re.IGNORECASE)
QUERY_MOBILE_DEVELOPMENT_RE = re.compile(
    r"Android\s*Studio|Android SDK|iOS SDK|SDK|API|开发|编译|源码|调试|adb|应用开发|系统开发|内核|驱动|命令行|配置文件",
    re.IGNORECASE,
)
QUERY_WINDOWS_RE = re.compile(r"Windows", re.IGNORECASE)
QUERY_WINDOWS_TECHNICAL_RE = re.compile(
    r"开发|编译|脚本|命令|配置|系统|服务|网络|错误|故障|排障|注册表|驱动|权限|安装|部署|服务器|PowerShell|cmd|DLL|tracert|启动|安全模式|日志|进程",
    re.IGNORECASE,
)
QUERY_SOFTWARE_COMPANION_RE = re.compile(
    r"软件|程序|开发|源码|编译|调试|数据库|命令行|API|SDK|IDE|Android Studio|前端|后端|版本|部署|服务端|客户端|网络|协议|域名|IP|代理|配置|安装|系统|日志|端口|权限|进程|虚拟机|镜像|语言|运行时|项目|工程|网页|HTML|JavaScript|Python|Java|Linux|Git|Docker|SQL|HTTP|TCP|DNS|服务器故障|排障|上网",
    re.IGNORECASE,
)
AMBIGUOUS_QUERY_ANCHOR_RE = re.compile(r"代码|依赖|框架|容器|脚本|编程|程序设计|调试|配置|服务器|网络|系统|软件|程序", re.IGNORECASE)
QUERY_NETWORK_RE = re.compile(r"网络", re.IGNORECASE)
QUERY_NETWORK_TECHNICAL_RE = re.compile(
    r"适配器|交换机|路由|协议|TCP|UDP|HTTP|DNS|IP|域名|端口|服务器|防火墙|代理|配置|故障|连接|布线|无线|WIFI|管理员|加密|抓包|带宽|链路|网络设置|上网|网络工程",
    re.IGNORECASE,
)
QUERY_WEAK_INFRA_RE = re.compile(
    r"网络|网络适配器|适配器|上网|代理|虚拟机|蓝牙|无线|热点|路由|路由器|交换机|布线|宽带|"
    r"(?<![A-Za-z0-9])(?:IP|DNS|HTTP|HTTPS|TCP(?:/IP)?|UDP|WIFI)(?![A-Za-z0-9])|"
    r"IP地址|域名|服务器|Windows|Win\s*\d+|开发者|浏览器|配置|安装包|端口|防火墙|网络管理员|网络连接|网络设置|服务端|客户端",
    re.IGNORECASE,
)
QUERY_EXPLICIT_SOFTWARE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:Python|Java(?:EE|SE|语言)?|JavaScript|C\+\+|C#|C语言|Go语言|Rust|PHP|Ruby|Kotlin|Swift|MATLAB|R语言|\.NET|Node(?:\.js)?|HTML|CSS|SQL|"
    r"MySQL|PostgreSQL|MongoDB|Redis|Linux|Unix|CentOS|Ubuntu|Git|Docker|Kubernetes|Nginx|Apache|Maven|Gradle|Spring(?:Boot|MVC)?|Django|Flask|React|Vue|Qt)(?![A-Za-z0-9])|"
    r"Android\s*(?:Studio|SDK)|iOS\s*SDK|软件开发|软件工程|程序设计|编程开发|源代码|源码|编译|调试(?:代码|程序)?|报错|错误排查|开发环境|开发工具|开发者工具|数据库|数据结构|命令行|Shell脚本|"
    r"(?<![A-Za-z0-9])(?:API|SDK|IDE)(?![A-Za-z0-9])|前端|后端|全栈|版本控制|包管理|Web开发|应用开发|接口调用|服务端部署|服务部署|服务器部署|服务器日志|服务器故障|服务器排障|"
    r"(?:系统|软件|数据库|服务器|服务端|客户端|防火墙|容器|虚拟机|网络服务|应用|开发环境|运行环境|项目).{0,4}配置|"
    r"配置.{0,4}(?:文件|防火墙|数据库|服务器|服务端|客户端|环境|参数|端口|权限|依赖|项目|应用|系统|软件)|"
    r"容器化|模型部署|DLL|注册表|PowerShell|(?<![A-Za-z0-9])cmd(?![A-Za-z0-9])|tracert",
    re.IGNORECASE,
)
PURE_CODE_TASK_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:Python|Java|C语言|C\+\+|C#|JavaScript|MATLAB|Go|PHP)(?![A-Za-z0-9]).{0,36}(?:统计.{0,12}(?:字符|个数|次数)|"
    r"相反次序|逆序|反向(?:次序|排序)?|重组|重新组合|(?:简单)?求和|累加|"
    r"输出.{0,12}(?:list|列表|内容)|输入.{0,12}(?:字符串|string)|反转.{0,12}链表|"
    r"递归.{0,12}(?:例子|示例)|(?:例子|示例).{0,12}递归)|"
    r"(?:反转.{0,12}链表|链表.{0,12}反转|统计.{0,12}(?:字符|个数|次数)|(?:简单)?求和|输出.{0,12}(?:list|列表|内容)|"
    r"输入.{0,12}(?:字符串|string)|递归.{0,12}(?:例子|示例)|(?:例子|示例).{0,12}递归).{0,36}"
    r"(?<![A-Za-z0-9])(?:Python|Java|C语言|C\+\+|C#|JavaScript|MATLAB|Go|PHP)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
QUERY_DEBUG_OR_EXPLANATION_RE = re.compile(
    r"哪错|错误|报错|异常|不合法|不正确|怎么回事|故障|排障|调试|debug|"
    r"函数.{0,8}(?:是什么|含义|作用|原理|用法|解释|区别)|(?:是什么|含义|作用|原理|用法|解释|区别).{0,8}函数",
    re.IGNORECASE,
)
QUERY_CONSUMER_RE = re.compile(
    r"微信视频|视频|播放|音乐|歌曲|电影|电视剧|账号|注册|刷机|升级|恢复出厂|来电|语音控制|手机|下载手机|装机|游戏|网游|手游|页游|购物|抢购",
    re.IGNORECASE,
)
PURE_CODE_QUERY_RE = re.compile(
    r"代码生成|生成代码|代码实例|代码示例|编程游戏|小程序怎么做|打印出.*代码|输出.*代码|"
    r"(?:编程|程序设计).*(?:做出|打印|输出|画出|编写|实现).*(?:图形|代码|程序|棱形|菱形|三角形|正方形)|"
    r"(?:写|编写|实现|输出|打印).*(?:代码|程序)|(?:代码|程序).*(?:例子|示例|求和|闰年|菱形|流水灯|两只老虎|贪吃蛇|正方体|体积|面积)|"
    r"(?:两只老虎|贪吃蛇).*(?:代码|C语言|Python)|九九乘法表|HTML.*(?:代码|怎么做|做)|(?:Python|Java|C语言|C\+\+|JavaScript).*(?:例子|示例)|Java.*程序.*(?:部门|企业)",
    re.IGNORECASE,
)
NON_SOFTWARE_RE = re.compile(
    r"数控|铣床|车床|加工中心|机械加工|股票|股票代码|证券|股市|基金|期货|外汇|彩票|教师资格|驾照|压力容器|医学|药品|美食|装修",
    re.IGNORECASE,
)
# Body admission requires a local software context, rather than a count of
# incidental network/server/code words.  English stack names are paired with
# a nearby technical action; database terms are paired with schema/query
# vocabulary; and system terms name an actual configuration or deployment
# operation.  These patterns are deliberately separate from the broad query
# anchors above.
BODY_STACK_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:Python|Java(?:EE|SE|语言)?|JavaScript|C\+\+|C#|C语言|Go语言|Rust|PHP|Ruby|Kotlin|Swift|MATLAB|R语言|\.NET|Node(?:\.js)?|HTML|CSS|SQL|"
    r"MySQL|PostgreSQL|MongoDB|Redis|Linux|Unix|CentOS|Ubuntu|Git|Docker|Kubernetes|Nginx|Apache|Maven|Gradle|Spring(?:Boot|MVC)?|Django|Flask|React|Vue|Qt|VSS|VS\s*Code|Visual\s*Studio|"
    r"Android\s*Studio|Android\s*SDK|iOS\s*SDK|PowerShell|Windows|nodeApplication)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
BODY_DATABASE_RE = re.compile(
    r"数据库|(?<![A-Za-z0-9])(?:MySQL|PostgreSQL|MongoDB|Redis|SQL)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
BODY_DATABASE_CONTEXT_RE = re.compile(
    r"数据表|表结构|主键|索引|查询语句|SQL语句|数据库脚本|连接数据库|字段|创建表|插入数据|执行查询|"
    r"DROP\s+TABLE|CREATE\s+TABLE|INSERT\s+INTO|SELECT.{0,40}FROM",
    re.IGNORECASE,
)
BODY_SOFTWARE_CONTEXT_RE = re.compile(
    r"软件开发|软件工程|程序设计|开发环境|调试(?:代码|程序|器|环境)|"
    r"数据结构|Shell脚本|版本控制|开源软件|软件包管理|包管理器|依赖包|Web开发|应用开发|接口调用|"
    r"配置文件|注册表|DLL|PowerShell|系统配置|系统排障|系统部署|服务配置|服务排障|服务部署|服务器部署|"
    r"服务端部署|容器化|模型部署|数据中心部署|SQL语句|查询语句|表结构|主键|签入|签出",
    re.IGNORECASE,
)
BODY_COMPILER_CONTEXT_RE = re.compile(
    r"(?:代码|程序|源码|项目|模块|语言|软件|环境|工具|函数|工程|cpp|C\+\+|链接|构建|生成).{0,12}编译(?:器|代码)?|"
    r"编译(?:器|代码)?.{0,12}(?:代码|程序|源码|项目|模块|语言|软件|环境|工具|函数|工程|cpp|C\+\+|链接|构建|生成)|"
    r"编译器.{0,80}(?:函数|代码|程序|源码|项目|工程|文件|模块)|"
    r"(?:函数|代码|程序|源码|项目|工程|文件|模块).{0,80}编译器",
    re.IGNORECASE,
)
BODY_DEV_TOOL_CONTEXT_RE = re.compile(
    r"(?:开发工具|开发者工具).{0,24}(?:代码|项目|编译|调试|语言|框架|IDE|版本|插件)|"
    r"(?:代码|项目|编译|调试|语言|框架|IDE|版本|插件).{0,24}(?:开发工具|开发者工具)",
    re.IGNORECASE,
)
BODY_COMMAND_CONTEXT_RE = re.compile(
    r"命令行.{0,24}(?:脚本|Shell|shell|参数|执行|终端|服务器|Linux|Unix|PowerShell|cmd|路径|权限|进程|服务)|"
    r"(?:脚本|Shell|shell|参数|执行|终端|服务器|Linux|Unix|PowerShell|cmd|路径|权限|进程|服务).{0,24}命令行",
    re.IGNORECASE,
)
BODY_WORKDIR_CONTEXT_RE = re.compile(
    r"工作目录.{0,24}(?:代码|项目|编译|仓库|签入|签出|源码|版本|脚本)|"
    r"(?:代码|项目|编译|仓库|签入|签出|源码|版本|脚本).{0,24}工作目录",
    re.IGNORECASE,
)
BODY_SOFTWARE_ACRONYM_CONTEXT_RE = re.compile(
    r"(?:(?<![A-Za-z0-9])(?:API|SDK|IDE)(?![A-Za-z0-9]).{0,24}(?:接口|调用|开发|代码|编程|调试|配置|工具|项目|集成|文档|函数|依赖)|"
    r"(?:接口|调用|开发|代码|编程|调试|配置|工具|项目|集成|文档|函数|依赖).{0,24}(?<![A-Za-z0-9])(?:API|SDK|IDE)(?![A-Za-z0-9]))",
    re.IGNORECASE,
)
BODY_PROGRAMMING_RE = re.compile(
    r"(?:编程|程序设计).{0,30}(?:代码|程序|开发|环境|工具|项目|语法|函数)|"
    r"(?:代码|程序|开发|环境|工具|项目|语法|函数).{0,30}(?:编程|程序设计)",
    re.IGNORECASE,
)
BODY_FRONT_BACK_RE = re.compile(
    r"(?:前端|后端).{0,20}(?:开发|代码|页面|框架|接口|JavaScript|HTML|CSS|API)|"
    r"(?:开发|代码|页面|框架|接口|JavaScript|HTML|CSS|API).{0,20}(?:前端|后端)",
    re.IGNORECASE,
)
BODY_STACK_CONTEXT_RE = re.compile(
    r"代码|开发|编译|调试|运行|安装|配置|部署|函数|脚本|测试|语法|属性|标签|元素|页面|框架|模块|方法|"
    r"命令|日志|接口|模型|依赖|包|版本|错误|异常|排障|源码|源代码|协议|表结构|主键|索引|事务|工作目录|"
    r"签入|签出|应用|服务端|数据库|查询|创建|插入|执行",
    re.IGNORECASE,
)
BODY_SYSTEM_CONTEXT_RE = re.compile(
    r"(?:软件|应用|系统|服务端|客户端|服务器|环境|项目).{0,4}(?:配置|部署)|"
    r"(?:配置|部署).{0,4}(?:软件|应用|系统|服务端|客户端|服务器|环境|项目)",
    re.IGNORECASE,
)
BODY_SOURCE_CONTEXT_RE = re.compile(
    r"(?:源代码|源码).{0,30}(?:项目|版本|编译|下载|获取|管理|工作目录|签入|签出|仓库|代码)|"
    r"(?:项目|版本|编译|下载|获取|管理|工作目录|签入|签出|仓库|代码).{0,30}(?:源代码|源码)",
    re.IGNORECASE,
)
BODY_SOURCE_NEGATIVE_RE = re.compile(r"(?:没有|无|不含|不包含|不提供|缺少).{0,20}(?:源代码|源码)", re.IGNORECASE)
BODY_DEPLOYMENT_RE = re.compile(
    r"系统配置|系统排障|系统部署|服务配置|服务排障|服务部署|服务器配置|服务器部署|服务端部署|"
    r"容器化|模型部署|数据中心部署|防火墙配置|注册表|PowerShell",
    re.IGNORECASE,
)
BODY_MOBILE_RE = re.compile(r"Android|安卓|iOS|iPhone|鸿蒙|苹果系统", re.IGNORECASE)
BODY_MOBILE_SOFTWARE_RE = re.compile(
    r"Android\s*(?:Studio|SDK)|iOS\s*SDK|应用开发|系统开发|内核开发|NDK|Gradle|Activity|Intent|Manifest|adb|移动端开发|手机应用开发",
    re.IGNORECASE,
)
BODY_MOBILE_CONSUMER_RE = re.compile(
    r"(?:手机|安卓|iPhone|小米|三星|华为|OnePlus|USB调试).{0,100}(?:连接|设置|调试|升级|刷机|装机|使用|点击|教程|步骤)|"
    r"(?:连接|设置|调试|升级|刷机|装机|使用|点击|教程|步骤).{0,100}(?:手机|安卓|iPhone|小米|三星|华为|OnePlus|USB调试)",
    re.IGNORECASE,
)
BODY_WINDOWS_RE = re.compile(r"Windows", re.IGNORECASE)
BODY_WINDOWS_CONTEXT_RE = re.compile(
    r"系统配置|系统排障|系统部署|服务配置|服务排障|注册表|PowerShell|cmd|DLL|驱动开发|日志|进程|"
    r"防火墙|端口|网络配置|网络连接|网络服务|入站规则|出站规则|netstat|telnet|命令行|终端|服务器",
    re.IGNORECASE,
)
BODY_WINDOWS_CONSUMER_RE = re.compile(r"重装系统|一键重装|装机|系统安装|恢复出厂|桌面使用|壁纸|播放器", re.IGNORECASE)
BODY_HARDWARE_RE = re.compile(
    r"HDMI|AV接口|DB9|M2接口|插头|剃须刀|信号定义|显示器|主板|硬盘|磁盘|显卡|SCSI|硬件|内存|插槽|千兆|网卡|"
    r"洗衣机|空调|电梯|热水器|遥控器|流水灯|PLC|伺服驱动器|扫描仪|打印机|摄像头|录像机|路由器|手机|汽车|"
    r"光驱|信号灯|指示灯|电气设备|LED芯片|电力设备|机床|机械设备",
    re.IGNORECASE,
)
BODY_HARDWARE_SOFTWARE_RE = re.compile(
    r"设备驱动开发|驱动程序开发|驱动开发|内核开发|嵌入式开发|硬件接口开发|"
    r"(?:(?:SDK|API).{0,20}(?:开发|编程|代码|集成|实现)|(?:开发|编程|代码|集成|实现).{0,20}(?:SDK|API))|"
    r"(?:Linux|Windows).{0,20}(?:驱动开发|内核开发|设备驱动)|编程.{0,20}(?:硬件|设备)|协议.{0,20}(?:代码|开发|实现)|配置文件.{0,20}(?:设备|驱动)",
    re.IGNORECASE,
)
# A few broad body vetoes are useful for consumer pages, but terms such as
# ``内存``/``课程``/``游戏`` also occur in genuine programming and systems
# documentation.  This evidence is intentionally split into independent
# semantic classes.  A repeated generic word (or a lone ``API``/``开发工具``)
# is not enough to exempt a body from a domain veto.
BODY_SOFTWARE_CODE_SEMANTIC_RE = re.compile(
    r"代码|源码|程序|函数|变量|语法|数组|指针|脚本|算法|数据结构|编程|编译|调试|表达式|类型|内存|"
    r"构造函数|析构函数|多线程|匿名函数|闭包|函数参数|返回值|变量作用域|静态成员|泛型|正则表达式|"
    r"AJAX|Ajax|异步回调|异常处理|内存管理|指针运算|系统调用|数据类型",
    re.IGNORECASE,
)
BODY_SOFTWARE_DESIGN_SEMANTIC_RE = re.compile(
    r"项目|模块|框架|接口|类(?:型|定义|方法|属性)|对象|方法|属性|标签|元素|协议|库函数|系统调用",
    re.IGNORECASE,
)
BODY_SOFTWARE_OPERATION_SEMANTIC_RE = re.compile(
    r"配置|部署|安装|测试|排障|日志|命令|终端|服务器|服务|权限|端口|进程|线程|文件|目录|连接|"
    r"依赖|版本|运行|执行|调用|创建|插入|更新|删除|读取|写入|查询|表结构|主键|索引",
    re.IGNORECASE,
)
BODY_SOFTWARE_CONCEPT_CONTEXT_RE = re.compile(
    r"构造函数|析构函数|多线程|匿名函数|闭包|函数参数|返回值|变量作用域|静态成员|泛型|正则表达式|"
    r"AJAX|Ajax|异步回调|异常处理|内存管理|指针运算|系统调用|数据类型",
    re.IGNORECASE,
)
BODY_PRODUCT_LISTING_RE = re.compile(
    r"软件大小|授权方式|下载次数|推荐星级|系统人气|产品参数|产品特点|服务热线|MD5\s*[:：]|镜像文件|"
    r"安装版|软件类型|免费版|安全检测|文件格式|更新时间|价格|购买|"
    r"产品概述|计费概述|产品支持|适合家庭|适合商铺|家庭和商铺|观看并录像|云服务套餐",
    re.IGNORECASE,
)
BODY_NONSOFTWARE_MARKETING_RE = re.compile(
    r"招聘|招聘公告|招聘信息|岗位|职位|职位要求|薪资|月薪|工作年限|本科以上|就业|人才需求|"
    r"职业生涯|培训机构|培训学校|找工作|求职|简历|offer",
    re.IGNORECASE,
)
BODY_SYSTEM_OPERATION_RE = re.compile(
    r"防火墙.{0,40}(?:端口|规则|配置|网络|服务|连接)|"
    r"(?:端口|规则).{0,40}(?:防火墙|开放|限制|配置|连接)|"
    r"(?:开放|限制).{0,30}端口|"
    r"(?:端口|防火墙).{0,30}(?:命令|终端|服务器|日志|排障)|"
    r"(?:网络配置|网络服务|入站规则|出站规则).{0,40}(?:端口|防火墙|连接|命令)",
    re.IGNORECASE,
)
BODY_PAGE_RE = re.compile(
    r"论文|学位|摘要库|课程|考试|试题|单选题|多选题|参考答案|在线作业|课程复习|招聘|岗位|招聘公告|律师|事务所|"
    r"保险|房产|税务|财务|健康运势|医院|医疗|驾驶|车辆|历史|古诗|诗词|身份证|银行|SWIFT|广告|"
    r"5G套餐|流量包|运营商|救护车|海参|网盘搜索|个人自我评价|"
    r"(?:录取|招生|成绩查询|志愿填报|学科评估|分数线|小题)|"
    r"(?:职位|岗位要求|薪资|月薪|工作年限|本科以上|面试)|"
    r"(?:诊所|医生|患者|癌症|手术|治疗)|"
    r"(?:研讨会|会议由|公共管理|移动政务|社交平台|创新2\.0|暗网)|"
    r"(?:低至\s*\d+折|产品特点|服务热线|了解详情|促销)|"
    r"(?:大学|学院).{0,30}(?:专业|排行榜|录取|分数线|招生)|"
    r"(?:专业|排行榜|录取|分数线|招生).{0,30}(?:大学|学院)|"
    r"(?:词典|词语大全|热门查询|查询的是)|"
    r"(?:悬赏|字母圈|调教|蝶恋花|译文|相思|王国维)|"
    r"(?:PKPM|AUTOCAD|红警|Game\.exe)",
    re.IGNORECASE,
)
BODY_CONSUMER_SOFTWARE_RE = re.compile(
    r"(?:Office(?:\s*365)?|办公软件|播放器|软件商店|应用商店|360浏览器|Chrome浏览器|QQ浏览器|微信客户端|"
    r"密码破解软件|破解软件|软件类型|网友评分|应用平台|免费软件|QQUpdate\.db|QQ彩信|互动空间|"
    r"SBuddy(?:Application|Base|Call|Friend)\.dll|WPS表格|怀旧大脚|大脚插件)",
    re.IGNORECASE,
)
BODY_AGGREGATE_NAV_RE = re.compile(
    r"(?:博文.*){2,}|(?:相关推荐|最新问题|热门查询|文章列表).*(?:相关推荐|最新问题|热门查询|文章列表)|"
    r"图书目录|章节目录|目录.{0,30}(?:第\s*\d+章|上册|下册)|(?:上册|下册).{0,30}目录",
    re.IGNORECASE,
)
BODY_SOFTWARE_PRODUCT_CATALOG_RE = re.compile(
    r"最受欢迎.{0,24}(?:工具|软件|产品)|选择适合.{0,24}(?:工具|软件|产品)|"
    r"(?:产品|软件).{0,24}(?:模板|应用场景|价格|计费|购买|优点|缺点)|"
    r"(?:工具|软件).{0,24}(?:产品相比|产品比较|选择推荐)",
    re.IGNORECASE,
)
BODY_BROWSER_CONSUMER_RE = re.compile(
    r"(?:(?:Chrome|Firefox|火狐)(?:浏览器)?.{0,60}(?:界面|首页|导航|插件|优点|缺点|推荐|下载|兼容性)|"
    r"浏览器.{0,60}(?:界面|首页|导航|插件|优点|缺点|推荐|下载|兼容性)|"
    r"(?:界面|首页|导航|插件|优点|缺点|推荐|下载|兼容性).{0,60}(?:Chrome|Firefox|火狐|浏览器))",
    re.IGNORECASE,
)
BODY_EDUCATION_RE = re.compile(r"编程猫|少儿编程|儿童编程|编程课程|编程培训|少儿.*编程|家长.*编程", re.IGNORECASE)
BODY_NAV_RE = re.compile(
    r"登录|注册|评论|点赞|收藏|VIP|播放|视频|首页|栏目|作者|责任编辑|点击|下一页|上一页|目录|导航|关注|分享|"
    r"广告|关于我们|联系我们|版权|悬赏|查看答案",
    re.IGNORECASE,
)
# This is retained only as an audit diagnostic.  Marker count is never used
# to prove that a body is code-dominated or to admit/reject a document.
CODE_MARKER_RE = re.compile(r"```|\b(?:import|from|def|class|function|return|public\s+static|#include)\b|[{};]|=>", re.IGNORECASE)


class PreparationBlocked(RuntimeError):
    """The requested audited slice cannot meet its explicit contract."""


@dataclass(frozen=True)
class QueryInfo:
    key: str
    source_split: str
    source_id: str
    text: str
    normalized: str
    template: str
    matches: tuple[str, ...]


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}
        self.rank = {key: 0 for key in self.parent}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _json_dump(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evidence file: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite evidence file: {path}")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False))
            stream.write("\n")


def normalize_body(text: str) -> str:
    """Normalize only Unicode and whitespace for lossless exact aliases."""

    if not isinstance(text, str):
        return ""
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).replace("\u3000", " ")).strip()


def normalize_query(text: str) -> str:
    value = normalize_body(text).casefold()
    return PUNCT_RE.sub("", value)


def query_template(text: str) -> str:
    value = normalize_query(text)
    return NUM_RE.sub("<num>", value)


def _technical_matches(text: str) -> tuple[str, ...]:
    return tuple(label for label, pattern in TECHNICAL_PATTERNS if pattern.search(text))


VERSIONED_QUERY_RE = re.compile(
    r"(?<![A-Za-z])(?:Python|Java|MySQL|PostgreSQL|CentOS|Ubuntu|Redis|Tomcat|PHP)"
    r"\s*\d+(?:[.]\d+)*(?:bit)?(?![A-Za-z0-9])", re.IGNORECASE,
)
DEVELOPMENT_TOOL_QUERY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:JDK|JVM|GitLab|GitHub|PyCharm|VSCode|IntelliJ|Tomcat|JDBC|"
    r"SQLite|SQLyog|PL/?SQL|Oracle(?:RAC)?|LinuxOracle|Hive|ProcessBuilder|ProcessBulider|"
    r"SSM|VBA|AJAX|Anaconda|Anoconda|Pandas|NumPy)(?![A-Za-z])|"
    r"(?<![A-Za-z0-9])IDEA(?![A-Za-z0-9]).{0,20}(?:版本|回退|替换|快捷键|project|运行|程序)|"
    r"(?:Python|Node|包管理|依赖包).{0,15}(?:pip|npm)|(?:pip|npm).{0,15}(?:Python|Node|依赖包)",
    re.IGNORECASE,
)
PROGRAMMING_CONCEPT_QUERY_RE = re.compile(
    r"构造函数|析构函数|虚函数|线程安全|线程不安全|线程池|匿名函数|闭包|函数参数|形参|实参|"
    r"返回值|变量作用域|静态成员|静态方法|泛型|函数指针|指针函数|空指针|智能指针|正则表达式|"
    r"指针.{0,12}(?:内存|队列|整型|数组)|(?:内存|队列|整型|数组).{0,12}指针|"
    r"HTTP状态码|HTTP请求|TCP连接|TCP三次握手|DNS解析", re.IGNORECASE,
)
QUERY_EXPLANATION_RE = re.compile(
    r"是什么|什么意思|为什么|作用|原理|区别|复杂度|报错|错误|异常|调试|不合法|不正确|被称为|正确吗",
    re.IGNORECASE,
)
TOY_PROGRAM_QUERY_RE = re.compile(
    r"斐波那契|杨辉三角|阶乘|最大公因数|最小公倍数|素数|完数|心形图案|"
    r"求.{0,10}(?:面积|体积|周长|n次方)|计算.{0,12}(?:提成|两点间距离)|"
    r"输出.{0,12}(?:以内|前\d+项)|打印.{0,12}(?:数列|图案)|"
    r"字符串向后移一位|实现.{0,40}题目|matlab解.{0,10}方程", re.IGNORECASE,
)
NONTECH_QUERY_RE = re.compile(
    r"学历要求|等级考试|二级.{0,8}不考|网络机顶盒|电视机|安装电脑操作系统|删除电脑的操作系统|"
    r"^\d{4}xp操作系统$|python下载多少钱|计算机二级|怎么复习|需要什么基础|"
    r"后端面试问.{0,20}还是|(?:CentOS|Ubuntu)\d*.{0,10}(?:截屏|截图)|"
    r"若a是int数量int变量|a什么意思c语言卜术事|^java语言程序设计[（(]一[）)]$|"
    r"^v如何对数据库进行优化$", re.IGNORECASE,
)


def classify_query(text: str) -> dict[str, Any]:
    """Return an auditable high precision software-query decision."""

    normalized = normalize_body(text)
    han_count = len(HAN_RE.findall(normalized))
    matches = _technical_matches(normalized)
    additional = tuple(label for label, pattern in (
        ('versioned_software', VERSIONED_QUERY_RE),
        ('development_tool', DEVELOPMENT_TOOL_QUERY_RE),
        ('programming_concept', PROGRAMMING_CONCEPT_QUERY_RE),
    ) if pattern.search(normalized))
    matches += additional
    explanation = bool(QUERY_DEBUG_OR_EXPLANATION_RE.search(normalized) or QUERY_EXPLANATION_RE.search(normalized))
    if not normalized:
        return {"pass": False, "reason": "empty_query", "matches": [], "han_count": 0}
    if han_count < 2:
        return {"pass": False, "reason": "not_meaningful_chinese", "matches": list(matches), "han_count": han_count}
    if QUERY_EXCLUSION_RE.search(normalized) or NONTECH_QUERY_RE.search(normalized):
        return {"pass": False, "reason": "non_software_context", "matches": list(matches), "han_count": han_count}
    if QUERY_CONSUMER_RE.search(normalized) or CONSUMER_RE.search(normalized):
        return {"pass": False, "reason": "consumer_or_game", "matches": list(matches), "han_count": han_count}
    if NON_SOFTWARE_RE.search(normalized):
        return {"pass": False, "reason": "non_software_technical_domain", "matches": list(matches), "han_count": han_count}
    if (PURE_CODE_QUERY_RE.search(normalized) or TOY_PROGRAM_QUERY_RE.search(normalized)) and not explanation:
        return {"pass": False, "reason": "pure_code_generation", "matches": list(matches), "han_count": han_count}
    if PURE_CODE_TASK_RE.search(normalized) and not explanation:
        return {"pass": False, "reason": "pure_code_task", "matches": list(matches), "han_count": han_count}
    if HARDWARE_RE.search(normalized) and not re.search(
        r"软件|系统|驱动|编程|代码|数据库|协议|网络|配置|开发|操作系统",
        normalized,
        re.IGNORECASE,
    ) and not additional:
        return {"pass": False, "reason": "consumer_hardware", "matches": list(matches), "han_count": han_count}
    if QUERY_MOBILE_RE.search(normalized) and not QUERY_MOBILE_DEVELOPMENT_RE.search(normalized):
        return {"pass": False, "reason": "consumer_mobile_use", "matches": list(matches), "han_count": han_count}
    if QUERY_WINDOWS_RE.search(normalized) and not QUERY_WINDOWS_TECHNICAL_RE.search(normalized):
        return {"pass": False, "reason": "generic_windows_use", "matches": list(matches), "han_count": han_count}
    if QUERY_WEAK_INFRA_RE.search(normalized) and not QUERY_EXPLICIT_SOFTWARE_RE.search(normalized) and not additional:
        return {"pass": False, "reason": "weak_infrastructure_without_software_context", "matches": list(matches), "han_count": han_count}
    if QUERY_NETWORK_RE.search(normalized) and not QUERY_NETWORK_TECHNICAL_RE.search(normalized):
        return {"pass": False, "reason": "generic_network_context", "matches": list(matches), "han_count": han_count}
    if not matches:
        return {"pass": False, "reason": "no_software_anchor", "matches": [], "han_count": han_count}
    if re.search(r"(?<![A-Za-z0-9])IDE(?![A-Za-z0-9])", normalized, re.IGNORECASE) and not re.search(
        r"开发|编程|代码|环境|Eclipse|Visual|Java|C\+\+|Python|软件",
        normalized,
        re.IGNORECASE,
    ):
        return {"pass": False, "reason": "ambiguous_software_anchor", "matches": list(matches), "han_count": han_count}
    if AMBIGUOUS_QUERY_ANCHOR_RE.search(normalized) and not QUERY_SOFTWARE_COMPANION_RE.search(normalized) and not additional:
        return {"pass": False, "reason": "ambiguous_software_anchor", "matches": list(matches), "han_count": han_count}
    return {"pass": True, "reason": "software_anchor", "matches": list(matches), "han_count": han_count}


def classify_document(text: str) -> dict[str, Any]:
    """Return a strict, auditable body decision; no model or similarity is used."""

    normalized = normalize_body(text)
    visible = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", normalized, flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<[^>]*>", " ", visible)
    visible = re.sub(r"(?:https?|ftp)://[^\s<>]+|www\.[^\s<>]+", " ", visible, flags=re.IGNORECASE)
    visible = SPACE_RE.sub(" ", visible).strip()
    matches = _technical_matches(visible)
    han_count = len(HAN_RE.findall(visible))
    content_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z0-9]", visible))
    chinese_ratio = han_count / max(1, content_chars)
    code_marker_chars = sum(len(match.group(0)) for match in CODE_MARKER_RE.finditer(visible))
    code_marker_fraction = code_marker_chars / max(1, content_chars)

    # Local context is the admission signal.  Broad technical matches remain
    # in evidence for review, but a lone network/server/IP term cannot pass.
    stack_context = any(
        BODY_STACK_CONTEXT_RE.search(visible[max(0, match.start() - 140): match.end() + 140])
        for match in BODY_STACK_RE.finditer(visible)
    )
    stack_terms = [match.group(0).casefold() for match in BODY_STACK_RE.finditer(visible)]
    non_windows_stack_context = any(
        term != "windows" and not term.startswith("windows ")
        for term in stack_terms
    )
    non_markup_stack_context = any(term not in {"html", "css", "windows"} for term in stack_terms)
    database_context = any(
        BODY_DATABASE_CONTEXT_RE.search(visible[max(0, match.start() - 160): match.end() + 160])
        for match in BODY_DATABASE_RE.finditer(visible)
    )
    compiler_context = bool(BODY_COMPILER_CONTEXT_RE.search(visible))
    dev_tool_context = bool(BODY_DEV_TOOL_CONTEXT_RE.search(visible))
    command_context = bool(BODY_COMMAND_CONTEXT_RE.search(visible))
    workdir_context = bool(BODY_WORKDIR_CONTEXT_RE.search(visible))
    software_concept_context = bool(BODY_SOFTWARE_CONCEPT_CONTEXT_RE.search(visible))
    software_context = bool(
        BODY_SOFTWARE_CONTEXT_RE.search(visible)
        or BODY_SOFTWARE_ACRONYM_CONTEXT_RE.search(visible)
        or compiler_context
        or dev_tool_context
        or command_context
        or workdir_context
        or software_concept_context
    )
    programming_context = bool(BODY_PROGRAMMING_RE.search(visible))
    front_back_context = bool(BODY_FRONT_BACK_RE.search(visible))
    system_context = bool(
        BODY_SYSTEM_CONTEXT_RE.search(visible)
        or BODY_DEPLOYMENT_RE.search(visible)
        or BODY_SYSTEM_OPERATION_RE.search(visible)
    )
    source_context = bool(BODY_SOURCE_CONTEXT_RE.search(visible) and not BODY_SOURCE_NEGATIVE_RE.search(visible))
    explicit_context = bool(
        database_context
        or (stack_context and not (BODY_WINDOWS_RE.search(visible) and not BODY_WINDOWS_CONTEXT_RE.search(visible)))
        or software_context
        or programming_context
        or front_back_context
        or system_context
        or source_context
    )
    strong_context = bool(database_context or software_context or front_back_context or system_context or source_context)
    semantic_classes = {
        "code": bool(BODY_SOFTWARE_CODE_SEMANTIC_RE.search(visible)),
        "design": bool(BODY_SOFTWARE_DESIGN_SEMANTIC_RE.search(visible)),
        "operation": bool(BODY_SOFTWARE_OPERATION_SEMANTIC_RE.search(visible)),
    }
    semantic_class_count = sum(semantic_classes.values())
    operation_terms = {
        match.group(0).casefold()
        for match in BODY_SOFTWARE_OPERATION_SEMANTIC_RE.finditer(visible)
    }
    operation_term_count = len(operation_terms)
    stack_install_context = bool(
        stack_context
        and operation_term_count >= 1
        and re.search(r"安装|配置|部署|编译|运行|版本", visible, re.IGNORECASE)
        and (
            non_markup_stack_context
            or (semantic_class_count >= 2 and operation_term_count >= 2)
        )
    )
    direct_software_context = bool(
        database_context
        or software_context
        or programming_context
        or front_back_context
        or system_context
        or source_context
    )
    product_listing = bool(BODY_PRODUCT_LISTING_RE.search(visible))
    marketing_terms = {
        match.group(0).casefold()
        for match in BODY_NONSOFTWARE_MARKETING_RE.finditer(visible)
    }
    marketing_page = bool(
        len(marketing_terms) >= 2
        or re.search(r"培训机构|培训学校|招聘公告|职位要求", visible, re.IGNORECASE)
    )
    # This is used only to exempt the three historically overbroad body
    # vetoes.  It requires two independent technical semantic classes and a
    # local software context; repeated words or a lone infrastructure term do
    # not qualify.  Stack-only evidence also needs an operation, so a bare
    # version/name mention cannot rescue a consumer page.
    software_content_context = bool(
        not product_listing
        and not marketing_page
        and (
            (
                database_context
                and semantic_class_count >= 1
            )
            or (
                direct_software_context
                and semantic_class_count >= 2
            )
            or (
                stack_context
                and semantic_class_count >= 2
                and semantic_classes["code"]
                and semantic_classes["operation"]
            )
            or (
                stack_context
                and operation_term_count >= 2
                and semantic_classes["operation"]
            )
            or stack_install_context
            or BODY_SYSTEM_OPERATION_RE.search(visible)
        )
    )
    nonsoftware_terms = {
        match.group(0).casefold()
        for match in NON_SOFTWARE_RE.finditer(visible)
    }
    incidental_nonsoftware = bool(
        software_content_context
        and len(nonsoftware_terms) == 1
        and semantic_class_count >= 2
        and (
            software_context
            or database_context
            or programming_context
            or front_back_context
            or system_context
            or source_context
            or len(stack_terms) >= 2
        )
    )
    # Hardware words require an explicit stack/database/system/software
    # anchor in addition to the semantic evidence.  This keeps generic API
    # and device marketing pages out while admitting C/Python/SQL/Linux
    # documentation that merely discusses memory, disk, or hardware.
    hardware_software_content_context = bool(
        software_content_context
        and (semantic_classes["code"] or semantic_classes["design"])
        and (
            stack_context
            or database_context
            or programming_context
            or front_back_context
            or system_context
            or source_context
        )
    )
    windows_technical_context = bool(
        BODY_WINDOWS_CONTEXT_RE.search(visible)
        or (
            software_content_context
            and (
                non_windows_stack_context
                or software_context
                or database_context
                or programming_context
                or front_back_context
                or system_context
                or source_context
            )
        )
    )
    # Generic action words such as 点击/目录 are common in software setup
    # tutorials.  Once the auditable software-content test has passed, they
    # do not by themselves turn the body into a navigation page.  Aggregate,
    # browser, product, and other structural vetoes still run earlier.
    strong_context = bool(strong_context or software_content_context)
    details = {
        "matches": list(matches),
        "han_count": han_count,
        "content_chars": content_chars,
        "chinese_ratio": round(chinese_ratio, 6),
        "code_marker_fraction": round(code_marker_fraction, 6),
        "stack_context": stack_context,
        "stack_terms": stack_terms,
        "non_windows_stack_context": non_windows_stack_context,
        "non_markup_stack_context": non_markup_stack_context,
        "database_context": database_context,
        "software_context": software_context,
        "compiler_context": compiler_context,
        "dev_tool_context": dev_tool_context,
        "command_context": command_context,
        "workdir_context": workdir_context,
        "software_concept_context": software_concept_context,
        "system_context": system_context,
        "source_context": source_context,
        "semantic_classes": semantic_classes,
        "semantic_class_count": semantic_class_count,
        "operation_terms": sorted(operation_terms),
        "operation_term_count": operation_term_count,
        "stack_install_context": stack_install_context,
        "marketing_terms": sorted(marketing_terms),
        "nonsoftware_terms": sorted(nonsoftware_terms),
        "incidental_nonsoftware": incidental_nonsoftware,
        "product_listing": product_listing,
        "marketing_page": marketing_page,
        "software_content_context": software_content_context,
        "hardware_software_content_context": hardware_software_content_context,
        "windows_technical_context": windows_technical_context,
    }
    if not normalized:
        return {"pass": False, "reason": "empty_body", **details}
    if len(normalized.encode("utf-8")) > 1_000_000:
        return {"pass": False, "reason": "document_too_large", **details}
    if han_count < 40 or content_chars < 80 or chinese_ratio < 0.35:
        return {"pass": False, "reason": "not_meaningful_chinese", **details}
    if BODY_EDUCATION_RE.search(visible):
        return {"pass": False, "reason": "consumer_or_game", **details}
    if CONSUMER_RE.search(visible) and not software_content_context:
        return {"pass": False, "reason": "consumer_or_game", **details}
    if NON_SOFTWARE_RE.search(visible) and not incidental_nonsoftware:
        return {"pass": False, "reason": "non_software_context", **details}
    if BODY_PAGE_RE.search(visible) and not software_content_context:
        return {"pass": False, "reason": "non_software_context", **details}
    if BODY_CONSUMER_SOFTWARE_RE.search(visible):
        return {"pass": False, "reason": "consumer_software_use", **details}
    if BODY_SOFTWARE_PRODUCT_CATALOG_RE.search(visible):
        return {"pass": False, "reason": "navigation_or_consumer_page", **details}
    # Strong technical anchors cannot rescue an aggregate page or a
    # consumer browser recommendation.  These are structural/content-domain
    # exclusions, so apply them before the strong-context admission shortcut.
    if BODY_AGGREGATE_NAV_RE.search(visible):
        return {"pass": False, "reason": "navigation_or_consumer_page", **details}
    if BODY_BROWSER_CONSUMER_RE.search(visible):
        return {"pass": False, "reason": "consumer_software_use", **details}
    if BODY_MOBILE_RE.search(visible) and not BODY_MOBILE_SOFTWARE_RE.search(visible):
        return {"pass": False, "reason": "consumer_mobile_use", **details}
    if BODY_MOBILE_CONSUMER_RE.search(visible) and not BODY_MOBILE_SOFTWARE_RE.search(visible):
        return {"pass": False, "reason": "consumer_mobile_use", **details}
    if BODY_WINDOWS_RE.search(visible) and not windows_technical_context:
        return {"pass": False, "reason": "generic_windows_use", **details}
    if BODY_WINDOWS_CONSUMER_RE.search(visible):
        return {"pass": False, "reason": "consumer_windows_use", **details}
    if BODY_HARDWARE_RE.search(visible) and not (
        BODY_HARDWARE_SOFTWARE_RE.search(visible) or hardware_software_content_context
    ):
        return {"pass": False, "reason": "consumer_hardware", **details}
    if product_listing:
        return {"pass": False, "reason": "non_software_context", **details}
    if not matches and not explicit_context:
        return {"pass": False, "reason": "no_software_anchor", **details}
    if not explicit_context:
        return {"pass": False, "reason": "no_explicit_software_context", **details}
    if BODY_NAV_RE.search(visible) and not strong_context:
        return {"pass": False, "reason": "navigation_or_consumer_page", **details}
    return {"pass": True, "reason": "software_context_and_meaningful_chinese", **details}


def _classify_document_worker(text: str) -> dict[str, Any]:
    """Process-pool entrypoint kept at module scope for spawn/fork parity."""

    return classify_document(text)


def _split_fields(line: str, minimum: int) -> list[str]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < minimum:
        fields = line.rstrip("\r\n").split()
    return fields


def _parse_grade(value: str) -> int | float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("grade is not finite")
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _read_queries(path: Path, source_split: str) -> tuple[dict[str, QueryInfo], list[dict[str, Any]], dict[str, Any]]:
    candidate: dict[str, QueryInfo] = {}
    decisions: list[dict[str, Any]] = []
    stats = {"path": str(path), "rows": 0, "nonempty": 0, "parsed": 0, "malformed": 0, "duplicates": 0}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stats["rows"] += 1
            if not line.strip():
                continue
            stats["nonempty"] += 1
            fields = _split_fields(line, 2)
            if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
                stats["malformed"] += 1
                decisions.append({"source_split": source_split, "line": line_number, "status": "excluded", "reason": "malformed_query_row"})
                continue
            source_id, text = fields[0].strip(), fields[1].strip()
            if source_id.casefold() in {"qid", "query_id", "query-id"} and text.casefold() in {"query", "text"}:
                continue
            stats["parsed"] += 1
            key = f"{source_split}:{source_id}"
            if key in candidate:
                stats["duplicates"] += 1
                decisions.append({"source_split": source_split, "source_id": source_id, "line": line_number, "status": "excluded", "reason": "duplicate_query_id"})
                continue
            classification = classify_query(text)
            decisions.append({
                "source_split": source_split,
                "source_id": source_id,
                "line": line_number,
                "status": "domain_candidate" if classification["pass"] else "excluded",
                "reason": classification["reason"],
                "matches": classification["matches"],
                "han_count": classification["han_count"],
            })
            if classification["pass"]:
                candidate[key] = QueryInfo(
                    key=key,
                    source_split=source_split,
                    source_id=source_id,
                    text=text,
                    normalized=normalize_query(text),
                    template=query_template(text),
                    matches=tuple(classification["matches"]),
                )
    stats["parse_rate"] = stats["parsed"] / stats["nonempty"] if stats["nonempty"] else 0.0
    return candidate, decisions, stats


def _read_qrels(
    path: Path,
    source_split: str,
    allowed_queries: Mapping[str, QueryInfo],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {"path": str(path), "rows": 0, "nonempty": 0, "parsed": 0, "stored": 0, "malformed": 0, "duplicate_pairs": 0}
    pair_seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stats["rows"] += 1
            if not line.strip():
                continue
            stats["nonempty"] += 1
            fields = _split_fields(line, 4)
            if len(fields) < 4:
                stats["malformed"] += 1
                continue
            source_id, document_id, grade_text = fields[0].strip(), fields[2].strip(), fields[3].strip()
            if source_id.casefold() in {"qid", "query_id", "query-id"}:
                continue
            try:
                grade = _parse_grade(grade_text)
            except ValueError:
                stats["malformed"] += 1
                continue
            stats["parsed"] += 1
            key = f"{source_split}:{source_id}"
            if key not in allowed_queries:
                continue
            pair = (key, document_id)
            if pair in pair_seen:
                stats["duplicate_pairs"] += 1
            pair_seen.add(pair)
            rows_by_query[key].append({
                "source_split": source_split,
                "source_id": source_id,
                "doc_id": document_id,
                "original_grade": grade,
                "line": line_number,
            })
            stats["stored"] += 1
    stats["parse_rate"] = stats["parsed"] / stats["nonempty"] if stats["nonempty"] else 0.0
    return dict(rows_by_query), stats


def _stream_collection(
    path: Path,
    positive_ids: set[str],
    *,
    workers: int = 1,
    batch_size: int = 1_000,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Stream a complete collection and retain only domain-valid bodies.

    With ``workers=1`` this follows the original serial path.  A worker count
    above one classifies bounded batches in a process pool, then applies every
    result in source order.  Applying results in order preserves duplicate-ID
    handling, rejection receipts, document insertion order, and the raw-byte
    SHA digest while avoiding an unbounded future queue for the full corpus.
    """

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    documents: dict[str, dict[str, Any]] = {}
    rejected_positive: dict[str, str] = {}
    stats: Counter[str] = Counter()
    digest = hashlib.sha256()

    def retain(document_id: str, text: str, classification: Mapping[str, Any]) -> None:
        if not classification["pass"]:
            stats[f"rejected:{classification['reason']}"] += 1
            if document_id in positive_ids:
                rejected_positive[document_id] = classification["reason"]
            return
        documents[document_id] = {
            "_id": document_id,
            "title": "",
            "text": text,
            "language": "zh",
            "metadata": {
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "source_split": "collection",
                "original_id": document_id,
                "annotation_source": "source",
                "native_language_evidence": "T2Ranking Sogou user search logs and Chinese source passage",
                "domain": "software",
                "translation_status": "native",
                "filter_version": FILTER_VERSION,
                "filter_matches": list(classification["matches"]),
                "filter_metrics": {
                    "content_chars": classification["content_chars"],
                    "chinese_ratio": classification["chinese_ratio"],
                    "stack_context": classification["stack_context"],
                    "database_context": classification["database_context"],
                    "software_context": classification["software_context"],
                    "system_context": classification["system_context"],
                    "source_context": classification["source_context"],
                },
            },
        }
        stats["retained"] += 1

    with Path(path).open("rb") as stream:
        if workers == 1:
            for raw_line in stream:
                stats["rows"] += 1
                digest.update(raw_line)
                if not raw_line.strip():
                    stats["blank"] += 1
                    continue
                stats["nonempty"] += 1
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    stats["malformed"] += 1
                    continue
                fields = line.rstrip("\r\n").split("\t", 1)
                if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                    stats["malformed"] += 1
                    continue
                document_id, text = fields[0].strip(), fields[1].strip()
                if document_id in documents:
                    stats["duplicate_ids"] += 1
                    if document_id in positive_ids and document_id not in rejected_positive:
                        rejected_positive[document_id] = "duplicate_document_id"
                    continue
                stats["parsed"] += 1
                retain(document_id, text, classify_document(text))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while True:
                    batch: list[tuple[str, str, Any | None]] = []
                    eof = False
                    for _ in range(batch_size):
                        raw_line = stream.readline()
                        if not raw_line:
                            eof = True
                            break
                        stats["rows"] += 1
                        digest.update(raw_line)
                        if not raw_line.strip():
                            stats["blank"] += 1
                            continue
                        stats["nonempty"] += 1
                        try:
                            line = raw_line.decode("utf-8")
                        except UnicodeDecodeError:
                            stats["malformed"] += 1
                            continue
                        fields = line.rstrip("\r\n").split("\t", 1)
                        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                            stats["malformed"] += 1
                            continue
                        document_id, text = fields[0].strip(), fields[1].strip()
                        future = None if document_id in documents else executor.submit(_classify_document_worker, text)
                        batch.append((document_id, text, future))
                    for document_id, text, future in batch:
                        classification = future.result() if future is not None else None
                        if document_id in documents:
                            stats["duplicate_ids"] += 1
                            if document_id in positive_ids and document_id not in rejected_positive:
                                rejected_positive[document_id] = "duplicate_document_id"
                            continue
                        stats["parsed"] += 1
                        retain(document_id, text, classification or classify_document(text))
                    if eof:
                        break
    stats["sha256"] = digest.hexdigest()
    stats["bytes"] = path.stat().st_size
    stats["parse_rate"] = stats["parsed"] / stats["nonempty"] if stats["nonempty"] else 0.0
    return documents, rejected_positive, dict(stats)


def _body_aliases(
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Collapse exact normalized bodies while retaining every source alias."""

    by_hash: dict[str, list[str]] = defaultdict(list)
    for document_id, row in documents.items():
        by_hash[_sha256_text(normalize_body(str(row["text"])))].append(document_id)
    canonical_by_source: dict[str, str] = {}
    canonical_documents: dict[str, dict[str, Any]] = {}
    alias_groups: dict[str, list[str]] = {}
    for body_hash, source_ids in sorted(by_hash.items()):
        aliases = sorted(source_ids)
        canonical_id = aliases[0]
        group_id = f"body-{body_hash[:20]}"
        for source_id in aliases:
            canonical_by_source[source_id] = canonical_id
        row = dict(documents[canonical_id])
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "canonical_document_id": canonical_id,
            "source_aliases": aliases,
            "source_alias_count": len(aliases),
            "alias_group_id": group_id,
        })
        row["metadata"] = metadata
        canonical_documents[canonical_id] = row
        if len(aliases) > 1:
            alias_groups[group_id] = aliases
    return canonical_by_source, canonical_documents, alias_groups


def evaluate_queries(
    queries: Mapping[str, QueryInfo],
    qrels: Mapping[str, list[Mapping[str, Any]]],
    documents: Mapping[str, Mapping[str, Any]],
    rejected_positive: Mapping[str, str],
    canonical_by_source: Mapping[str, str] | None = None,
    *,
    max_positive: int = 200,
    alias_groups: Mapping[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Map source judgments to canonical docs and enforce whole-query retention."""

    eligible: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    canonical_by_source = canonical_by_source or {document_id: document_id for document_id in documents}
    alias_by_document = {
        document_id: group_id
        for group_id, source_ids in (alias_groups or {}).items()
        for document_id in source_ids
    }
    for key, query in sorted(queries.items()):
        rows = list(qrels.get(key, []))
        mapped_rows: list[dict[str, Any]] = []
        grades: dict[str, list[Any]] = defaultdict(list)
        source_ids_by_grade_key: defaultdict[str, set[str]] = defaultdict(set)
        source_grades_by_grade_key: defaultdict[str, defaultdict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            source_doc_id = str(row["doc_id"])
            canonical_id = canonical_by_source.get(source_doc_id)
            mapped = dict(row)
            mapped.update({
                "source_doc_id": source_doc_id,
                "canonical_doc_id": canonical_id,
                "mapping_status": "mapped" if canonical_id in documents else "unavailable",
            })
            mapped_rows.append(mapped)
            grade_key = canonical_id if canonical_id is not None else source_doc_id
            grades[grade_key].append(row["original_grade"])
            source_ids_by_grade_key[grade_key].add(source_doc_id)
            source_grades_by_grade_key[grade_key][source_doc_id].append(row["original_grade"])
        positive_ids = sorted(doc_id for doc_id, values in grades.items() if any(float(value) >= 2 for value in values))
        reasons: list[str] = []
        if not rows:
            reasons.append("no_qrels")
        if not positive_ids:
            reasons.append("no_grade_ge_2_positive")
        if len(positive_ids) > max_positive:
            reasons.append("positive_count_over_limit")
        unavailable: list[dict[str, Any]] = []
        for document_id in positive_ids:
            if document_id in documents:
                continue
            unavailable.append({
                "doc_id": document_id,
                "source_doc_ids": sorted(source_ids_by_grade_key[document_id]),
                "source_grades": {
                    source_doc_id: values
                    for source_doc_id, values in sorted(source_grades_by_grade_key[document_id].items())
                },
                "reason": rejected_positive.get(document_id, "missing_from_collection"),
            })
        if unavailable:
            reasons.append("positive_unusable_whole_query")
        decision = {
            "query_key": key,
            "source_split": query.source_split,
            "source_id": query.source_id,
            "text": query.text,
            "positive_count": len(positive_ids),
            "positive_ids": positive_ids,
            "unavailable_positive": unavailable,
            "reasons": sorted(set(reasons)),
            "status": "retained" if not reasons else "excluded",
        }
        decisions.append(decision)
        if reasons:
            continue
        max_grades = {doc_id: max(values, key=float) for doc_id, values in grades.items() if doc_id in documents}
        eligible[key] = {
            "query": query,
            "judgments": rows,
            "mapped_judgments": mapped_rows,
            "grades": max_grades,
            "positive_ids": positive_ids,
        }
        for mapped in mapped_rows:
            if float(mapped["original_grade"]) < 2:
                continue
            source_doc_id = mapped["source_doc_id"]
            canonical_id = mapped["canonical_doc_id"]
            retention.append({
                "query_key": key,
                "source_split": query.source_split,
                "source_query_id": query.source_id,
                "source_doc_id": source_doc_id,
                "canonical_doc_id": canonical_id,
                "original_grade": mapped["original_grade"],
                "mapping_status": mapped["mapping_status"],
                "retained": canonical_id in documents,
                "binary_qrel": 1 if canonical_id in documents else None,
                "alias_group_id": alias_by_document.get(source_doc_id),
            })
    return eligible, decisions, retention


def group_queries(
    eligible: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Connect shared positives, exact normalized queries, templates, aliases."""

    keys = sorted(eligible)
    uf = _UnionFind(keys)
    token_members: dict[str, list[str]] = defaultdict(list)
    alias_by_doc = {
        document_id: str((row.get("metadata") or {}).get("alias_group_id"))
        for document_id, row in documents.items()
        if (row.get("metadata") or {}).get("alias_group_id")
    }
    positive_members: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        query = eligible[key]["query"]
        token_members[f"normalized:{query.normalized}"].append(key)
        token_members[f"template:{query.template}"].append(key)
        for document_id in eligible[key]["positive_ids"]:
            positive_members[document_id].append(key)
            alias_group = alias_by_doc.get(document_id)
            if alias_group:
                token_members[f"positive_alias:{alias_group}"].append(key)
    for token, members in token_members.items():
        for left, right in zip(members, members[1:]):
            uf.union(left, right)
    for document_id, members in positive_members.items():
        for left, right in zip(members, members[1:]):
            uf.union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        components[uf.find(key)].append(key)
    groups: dict[str, list[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for members in sorted((sorted(values) for values in components.values()), key=lambda rows: rows):
        group_id = f"query-group-{_sha256_text('|'.join(members))[:20]}"
        groups[group_id] = members
        group_reason: dict[str, list[str]] = defaultdict(list)
        # Recompute token evidence from the final component, so union order
        # cannot hide a reason that was attached before a later union.
        for key in members:
            query = eligible[key]["query"]
            group_reason["normalized_query"].append(query.normalized)
            group_reason["query_template"].append(query.template)
            for document_id in eligible[key]["positive_ids"]:
                group_reason["shared_positive"].append(document_id)
                alias_group = alias_by_doc.get(document_id)
                if alias_group:
                    group_reason["positive_alias"].append(alias_group)
        metadata[group_id] = {
            "query_ids": members,
            "size": len(members),
            "reasons": {name: sorted(set(values)) for name, values in sorted(group_reason.items())},
        }
    return groups, metadata


def _preview_query_splits(records: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Call the benchmark-v2 sampler used by the final suite."""

    return sample_query_splits(
        records,
        dev_size=10,
        test_size=50,
        seed=42,
        task_id="software_zh",
        group_key="group_id",
    )


def _file_receipt(path: Path, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    result = {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}
    if expected:
        result["expected"] = {key: expected[key] for key in ("bytes", "sha256") if key in expected}
        mismatches = []
        if "bytes" in expected and int(expected["bytes"]) != size:
            mismatches.append("bytes")
        if "sha256" in expected and str(expected["sha256"]) != result["sha256"]:
            mismatches.append("sha256")
        result["mismatches"] = mismatches
    return result


def _read_json_if(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else None


def _expected_small_files(raw_root: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    # The complete acquisition receipt is applied last so it supersedes
    # split-specific receipts when a later download records both dev/train.
    for receipt_name in (
        "small-files-receipt.json",
        "train-acquisition-receipt.json",
        "acquisition-receipt.json",
    ):
        receipt = _read_json_if(Path(raw_root) / receipt_name)
        if not receipt:
            continue
        for item in receipt.get("files", []):
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                result[item["path"]] = item
    return result


def _write_blocked(evidence_dir: Path, reason: str, details: Mapping[str, Any] | None = None) -> None:
    _json_dump(evidence_dir / "status.json", {
        "status": "blocked",
        "reason": reason,
        "details": dict(details or {}),
        "created_at": _utc_now(),
        "gold_generation": False,
    })


def _check_paths(raw_root: Path, output: Path, evidence_dir: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    raw_root, output, evidence_dir = Path(raw_root), Path(output), Path(evidence_dir)
    output_real, evidence_real = output.resolve(), evidence_dir.resolve()
    if output_real == evidence_real or output_real in evidence_real.parents or evidence_real in output_real.parents:
        raise ValueError("output and evidence-dir must be disjoint")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new release directory: {output}")
    data_root = raw_root / "data"
    collection = data_root / "collection.tsv"
    if collection.name.endswith(".part") or not collection.is_file():
        raise PreparationBlocked(
            f"complete data/collection.tsv is required; refusing to read or rename {data_root / 'collection.tsv.part'}"
        )
    paths = {"collection": collection}
    missing: list[str] = []
    for source_split in splits:
        query_path, qrels_path = data_root / f"queries.{source_split}.tsv", data_root / f"qrels.{source_split}.tsv"
        if not query_path.is_file():
            missing.append(str(query_path))
        if not qrels_path.is_file():
            missing.append(str(qrels_path))
        paths[f"queries.{source_split}"] = query_path
        paths[f"qrels.{source_split}"] = qrels_path
    if missing:
        raise PreparationBlocked("requested original split files are missing: " + ", ".join(missing))
    return paths


def prepare_release(
    raw_root: Path,
    output: Path,
    evidence_dir: Path,
    *,
    splits: tuple[str, ...] = ("dev", "train"),
    minimum_queries: int = 60,
    minimum_documents: int = 1_000,
    workers: int = 1,
) -> dict[str, Any]:
    """Build, audit, and publish a new release, or write a blocking receipt."""

    raw_root, output, evidence_dir = Path(raw_root), Path(output), Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = _check_paths(raw_root, output, evidence_dir, splits)
    except PreparationBlocked as exc:
        _write_blocked(evidence_dir, str(exc))
        raise
    if tuple(dict.fromkeys(splits)) != splits:
        raise ValueError("splits must be unique and ordered")
    unknown = sorted(set(splits) - {"dev", "train"})
    if unknown:
        raise ValueError(f"unsupported original split(s): {unknown}")

    expected_small = _expected_small_files(raw_root)
    source_files: list[dict[str, Any]] = []
    for source_split in splits:
        for name in (f"queries.{source_split}.tsv", f"qrels.{source_split}.tsv"):
            relative = f"data/{name}"
            path_key = f"queries.{source_split}" if name.startswith("queries.") else f"qrels.{source_split}"
            source_files.append(_file_receipt(paths[path_key], expected_small.get(relative)))
    for receipt in source_files:
        if receipt.get("mismatches"):
            _write_blocked(evidence_dir, "small source file hash/size mismatch", {"file": receipt})
            raise PreparationBlocked(f"small source file does not match receipt: {receipt['path']}")

    all_queries: dict[str, QueryInfo] = {}
    query_decisions: list[dict[str, Any]] = []
    query_parse: dict[str, Any] = {}
    all_qrels: dict[str, list[dict[str, Any]]] = {}
    qrel_parse: dict[str, Any] = {}
    for source_split in splits:
        queries, decisions, stats = _read_queries(paths[f"queries.{source_split}"], source_split)
        all_queries.update(queries)
        query_decisions.extend(decisions)
        query_parse[source_split] = stats
        qrels, stats = _read_qrels(paths[f"qrels.{source_split}"], source_split, queries)
        all_qrels.update(qrels)
        qrel_parse[source_split] = stats

    positive_ids = {
        str(row["doc_id"])
        for rows in all_qrels.values()
        for row in rows
        if float(row["original_grade"]) >= 2
    }
    try:
        documents, rejected_positive, collection_stats = _stream_collection(
            paths["collection"], positive_ids, workers=workers
        )
    except OSError as exc:
        _write_blocked(evidence_dir, "cannot stream complete collection", {"error": str(exc)})
        raise PreparationBlocked(str(exc)) from exc
    expected_collection = {"bytes": COLLECTION_BYTES, "sha256": COLLECTION_LFS_SHA256}
    mismatch = []
    if int(expected_collection["bytes"]) != collection_stats["bytes"]:
        mismatch.append("bytes")
    if str(expected_collection["sha256"]) != collection_stats["sha256"]:
        mismatch.append("sha256")
    if mismatch:
        _write_blocked(evidence_dir, "collection does not match pinned upstream receipt", {
            "expected": expected_collection,
            "actual": {"bytes": collection_stats["bytes"], "sha256": collection_stats["sha256"]},
            "mismatches": mismatch,
        })
        raise PreparationBlocked("collection.tsv does not match pinned upstream size/SHA256")
    source_documents = documents
    canonical_by_source, documents, alias_groups = _body_aliases(source_documents)
    eligible, eligibility_decisions, retention = evaluate_queries(
        all_queries,
        all_qrels,
        documents,
        rejected_positive,
        canonical_by_source,
        max_positive=POLICY["max_positive_per_query"],
        alias_groups=alias_groups,
    )
    groups, group_metadata = group_queries(eligible, documents)
    if len(documents) < minimum_documents:
        _json_dump(evidence_dir / "domain_rules.json", _domain_rules())
        _json_dump(evidence_dir / "parse_receipt.json", _parse_receipt(source_files, query_parse, qrel_parse, collection_stats))
        _json_dump(evidence_dir / "selection.json", {"status": "blocked", "reason": "fewer_than_minimum_documents", "counts": {"documents": len(documents), "queries": len(eligible)}, "minimum_documents": minimum_documents})
        _jsonl_dump(evidence_dir / "query_exclusions.jsonl", query_decisions + eligibility_decisions)
        _write_blocked(evidence_dir, "fewer than the required distinct valid software documents", {"documents": len(documents), "minimum_documents": minimum_documents})
        raise PreparationBlocked(f"software domain has {len(documents)} valid documents; minimum is {minimum_documents}")
    if len(eligible) < minimum_queries:
        _json_dump(evidence_dir / "domain_rules.json", _domain_rules())
        _json_dump(evidence_dir / "parse_receipt.json", _parse_receipt(source_files, query_parse, qrel_parse, collection_stats))
        _json_dump(evidence_dir / "selection.json", {"status": "blocked", "reason": "fewer_than_minimum_queries", "counts": {"documents": len(documents), "queries": len(eligible)}, "minimum_queries": minimum_queries})
        _jsonl_dump(evidence_dir / "query_exclusions.jsonl", query_decisions + eligibility_decisions)
        _write_blocked(evidence_dir, "fewer than the required retained software queries", {"queries": len(eligible), "minimum_queries": minimum_queries})
        raise PreparationBlocked(f"software domain has {len(eligible)} retained queries; minimum is {minimum_queries}")
    # IDs remain original whenever they are unique across requested source
    # splits.  A collision receives a split namespace while metadata keeps the
    # original query ID and split explicit.
    by_source_id: defaultdict[str, list[str]] = defaultdict(list)
    for key in eligible:
        by_source_id[eligible[key]["query"].source_id].append(key)
    output_id: dict[str, str] = {}
    for key in eligible:
        query = eligible[key]["query"]
        output_id[key] = query.source_id if len(by_source_id[query.source_id]) == 1 else f"{query.source_split}:{query.source_id}"
    group_by_key = {
        key: group_id for group_id, members in groups.items() for key in members
    }
    preview_records = [
        {"query_id": output_id[key], "group_id": group_by_key[key]}
        for key in sorted(eligible)
    ]
    try:
        preview = _preview_query_splits(preview_records)
    except (SamplingShortageError, ValueError) as exc:
        _json_dump(evidence_dir / "domain_rules.json", _domain_rules())
        _json_dump(evidence_dir / "parse_receipt.json", _parse_receipt(source_files, query_parse, qrel_parse, collection_stats))
        _json_dump(evidence_dir / "selection.json", {
            "status": "blocked",
            "reason": str(exc),
            "counts": {"documents": len(documents), "queries": len(eligible), "groups": len(groups)},
            "sampling": {"seed": 42, "task_id": "software_zh", "entrypoint": "dci_bench.data.benchmark_v2.sample_query_splits"},
        })
        _jsonl_dump(evidence_dir / "query_exclusions.jsonl", query_decisions + eligibility_decisions)
        _write_blocked(evidence_dir, str(exc), {"groups": len(groups), "eligible_queries": len(eligible)})
        raise PreparationBlocked(str(exc)) from exc
    preview_by_id = {
        query_id: split
        for split in ("dev", "test")
        for query_id in preview[split]
    }
    preview_smoke_ids = set(preview["smoke"])
    final_queries: list[dict[str, Any]] = []
    final_qrels: dict[str, dict[str, int]] = {}
    final_judgments: list[dict[str, Any]] = []
    original_judgments: list[dict[str, Any]] = []
    for key in sorted(all_qrels):
        query = all_queries[key]
        for row in sorted(all_qrels[key], key=lambda value: (str(value["doc_id"]), int(value["line"]))):
            source_doc_id = str(row["doc_id"])
            canonical_id = canonical_by_source.get(source_doc_id)
            original_judgments.append({
                "query_key": key,
                "query_id": output_id.get(key),
                "original_query_id": row["source_id"],
                "original_split": query.source_split,
                "source_doc_id": source_doc_id,
                "canonical_doc_id": canonical_id,
                "original_grade": row["original_grade"],
                "source_line": row["line"],
                "mapping_status": "mapped" if canonical_id in documents else "unavailable",
                "eligible_query": key in eligible,
                "document_in_release": canonical_id in documents,
            })
    for key in sorted(eligible, key=lambda value: output_id[value]):
        item = eligible[key]
        query = item["query"]
        query_id = output_id[key]
        final_queries.append({
            "_id": query_id,
            "text": query.text,
            "language": "zh",
            "metadata": {
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "source_split": query.source_split,
                "original_split": query.source_split,
                "original_query_id": query.source_id,
                "group_id": group_by_key[key],
                "normalized_query": query.normalized,
                "query_template": query.template,
                "annotation_source": "source",
                "translation_status": "native",
                "native_language_evidence": "T2Ranking Sogou user search log; query retained as Chinese source text",
                "domain": "software",
                "filter_version": FILTER_VERSION,
                "sampling_preview_split": preview_by_id.get(query_id),
                "sampling_preview_smoke": query_id in preview_smoke_ids,
                "positive_original_grades": {doc_id: item["grades"][doc_id] for doc_id in item["positive_ids"]},
            },
        })
        final_qrels[query_id] = {doc_id: 1 for doc_id in item["positive_ids"]}
        for row in sorted(item["mapped_judgments"], key=lambda value: (str(value["source_doc_id"]), int(value["line"]))):
            canonical_id = row["canonical_doc_id"]
            if canonical_id not in documents:
                continue
            final_judgments.append({
                "query_id": query_id,
                "original_query_id": row["source_id"],
                "original_split": row["source_split"],
                "source_doc_id": row["source_doc_id"],
                "canonical_doc_id": canonical_id,
                "original_grade": row["original_grade"],
                "source_line": row["line"],
                "mapping_status": row["mapping_status"],
                "doc_id": canonical_id,
                "score": row["original_grade"],
                "status": "judged",
                "source": "human",
                "annotation_source": "human",
                "document_in_release": True,
            })

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "input_kind": "retrieval",
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "resolved_revision": REVISION,
            "connector": "t2ranking_tsv",
            "license": LICENSE,
            "source_urls": SOURCE_URLS,
        },
        "documents": [documents[document_id] for document_id in sorted(documents)],
        "queries": final_queries,
        "qrels": final_qrels,
        "evidence": [],
        "judgments": final_judgments,
        "provenance": {
            "source_id": PROVENANCE_SOURCE_ID,
            "dataset": DATASET,
            "revision": REVISION,
            "license": LICENSE,
            "domain": "software",
            "language": "zh",
            "translation_status": "native",
            "native_language_evidence": "T2Ranking paper documents Sogou user search logs; retained bodies and queries pass Chinese-language checks",
            "original_splits": list(splits),
            "selection_policy": {
                "filter_version": FILTER_VERSION,
                "filter_workers": workers,
                "all_grade_ge_2_positives_required": True,
                "grade_threshold": 2,
                "qrels_output": "binary_1_only_for_source_grade_ge_2",
                "original_grades_file": "judgments.jsonl",
                "group_isolation": "shared_positive_connected_components_plus_normalized_query_and_template_plus_exact_body_alias",
                "publish_policy": POLICY,
            },
        },
        # The core publisher also emits its generic train/validation/test
        # component report.  This explicit map is the protocol selection used
        # by this source preparation and records the required 10/50 DCI split.
        "split_assignments": {
            "version": "t2ranking-software-zh-preview-v1",
            "queries": preview,
            "counts": {split: len(preview[split]) for split in ("dev", "test", "smoke")},
            "seed": 42,
            "task_id": "software_zh",
            "group_key": "group_id",
            "entrypoint": "dci_bench.data.benchmark_v2.sample_query_splits",
        },
    }
    _json_dump(evidence_dir / "domain_rules.json", _domain_rules())
    _json_dump(evidence_dir / "parse_receipt.json", _parse_receipt(source_files, query_parse, qrel_parse, collection_stats))
    _json_dump(evidence_dir / "source_provenance.json", {
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "source_urls": SOURCE_URLS,
        "source_files": source_files + [{"path": str(paths["collection"]), "bytes": collection_stats["bytes"], "sha256": collection_stats["sha256"], "expected": expected_collection}],
        "document_filter_counts": {
            key: value for key, value in sorted(collection_stats.items()) if key.startswith("rejected:")
        },
        "positive_document_rejections": dict(sorted(rejected_positive.items())),
        "official_collection_lfs_oid": COLLECTION_LFS_SHA256,
        "official_collection_bytes": COLLECTION_BYTES,
        "canonical_document_count": len(documents),
        "source_document_count": len(source_documents),
        "exact_alias_document_count": len(source_documents) - len(documents),
        "alias_groups": {group_id: aliases for group_id, aliases in sorted(alias_groups.items())},
    })
    _json_dump(evidence_dir / "selection.json", {
        "status": "planned",
        "counts": {"candidate_documents": len(documents), "source_documents": len(source_documents), "exact_alias_documents_removed": len(source_documents) - len(documents), "eligible_queries": len(eligible), "query_groups": len(groups), "preview_queries": len(preview["dev"]) + len(preview["test"]), "preview_dev": len(preview["dev"]), "preview_test": len(preview["test"]), "preview_smoke": len(preview["smoke"]), "positive_pairs": sum(len(item["positive_ids"]) for item in eligible.values())},
        "minimums": {"documents": minimum_documents, "queries": minimum_queries},
        "original_splits": list(splits),
        "sampling": {"seed": 42, "task_id": "software_zh", "group_key": "group_id", "entrypoint": "dci_bench.data.benchmark_v2.sample_query_splits", "preview": preview},
        "policy": POLICY,
        "gold_generation": False,
    })
    _jsonl_dump(evidence_dir / "query_exclusions.jsonl", query_decisions + eligibility_decisions)
    _jsonl_dump(evidence_dir / "original_judgments.jsonl", original_judgments)
    _jsonl_dump(evidence_dir / "positive_retention.jsonl", [row | {"output_query_id": output_id[row["query_key"]]} for row in retention])
    _json_dump(evidence_dir / "query_groups.json", {"groups": group_metadata, "sampling_preview": preview, "group_key": "group_id"})

    try:
        manifest = publish_bundle(
            bundle,
            output,
            specification={
                "id": PROVENANCE_SOURCE_ID,
                "connector": "t2ranking_tsv",
                "input_kind": "retrieval",
                "revision": REVISION,
                "options": {
                    "dataset": DATASET,
                    "splits": list(splits),
                    "filter_version": FILTER_VERSION,
                    "filter_workers": workers,
                },
                "domain": "software",
                "language": "zh",
            },
            policy=POLICY,
        )
    except Exception as exc:
        _write_blocked(evidence_dir, "publish_bundle rejected the prepared data", {"error": type(exc).__name__ + ": " + str(exc)})
        raise
    try:
        published_bundle = load_bundle(output)
        published_manifest = verify_release(output)
        audited_fields = ("documents", "queries", "qrels", "evidence", "judgments")
        mismatches = [field for field in audited_fields if published_bundle.get(field) != bundle.get(field)]
        if mismatches:
            raise ValueError(f"published bundle changed audited inputs: {mismatches}")
        if published_manifest.get("state") != "labels_ready":
            raise ValueError(f"published release is not labels_ready: {published_manifest.get('state')}")
    except Exception as exc:
        _write_blocked(evidence_dir, "published release round-trip audit failed", {
            "error": type(exc).__name__ + ": " + str(exc),
        })
        raise PreparationBlocked(str(exc)) from exc
    _json_dump(evidence_dir / "status.json", {
        "status": "published",
        "output": str(output),
        "manifest": manifest,
        "published_state": published_manifest["state"],
        "round_trip_audit": {"load_bundle": True, "verify_release": True, "audited_fields_equal": True},
        "counts": {"documents": len(documents), "source_documents": len(source_documents), "queries": len(final_queries), "positive_pairs": sum(len(row) for row in final_qrels.values()), "judgments": len(final_judgments), "original_judgments": len(original_judgments)},
        "created_at": _utc_now(),
        "gold_generation": False,
    })
    return manifest


def _domain_rules() -> dict[str, Any]:
    return {
        "version": FILTER_VERSION,
        "language": "zh",
        "include": {
            "query": "at least two Han characters, a bounded software/runtime/stack anchor or an explicit development/configuration/deployment context; weak network, IP, DNS, domain, desktop, browser, server, or install-package triggers require that explicit context",
            "document": "visible body after removing HTML and URLs must contain at least 40 Han characters, 80 Han/alphanumeric characters, and Han ratio >=0.35; admission then requires local stack-plus-action, database-plus-schema/query, or explicit software development/system configuration/deployment evidence",
            "technical_anchor_groups": {label: pattern.pattern for label, pattern in TECHNICAL_PATTERNS},
        },
        "exclude": {
            "consumer_or_game": CONSUMER_RE.pattern,
            "query_consumer_or_game": QUERY_CONSUMER_RE.pattern,
            "query_non_software_context": QUERY_EXCLUSION_RE.pattern,
            "query_network_context_required": QUERY_NETWORK_TECHNICAL_RE.pattern,
            "query_weak_infrastructure_requires_explicit_software": QUERY_WEAK_INFRA_RE.pattern,
            "query_explicit_software_context": QUERY_EXPLICIT_SOFTWARE_RE.pattern,
            "query_mobile_requires_development": QUERY_MOBILE_DEVELOPMENT_RE.pattern,
            "query_windows_requires_technical_context": QUERY_WINDOWS_TECHNICAL_RE.pattern,
            "query_ambiguous_anchor_requires_companion": QUERY_SOFTWARE_COMPANION_RE.pattern,
            "hardware_without_software_context": HARDWARE_RE.pattern,
            "non_software_technical_domain": NON_SOFTWARE_RE.pattern,
            "pure_code_generation_query": PURE_CODE_QUERY_RE.pattern,
            "pure_code_task": PURE_CODE_TASK_RE.pattern,
            "debug_or_explanation_exception": QUERY_DEBUG_OR_EXPLANATION_RE.pattern,
            "document_stack": BODY_STACK_RE.pattern,
            "document_database": BODY_DATABASE_RE.pattern,
            "document_database_context": BODY_DATABASE_CONTEXT_RE.pattern,
            "document_software_context": BODY_SOFTWARE_CONTEXT_RE.pattern,
            "document_compiler_context": BODY_COMPILER_CONTEXT_RE.pattern,
            "document_developer_tool_context": BODY_DEV_TOOL_CONTEXT_RE.pattern,
            "document_command_context": BODY_COMMAND_CONTEXT_RE.pattern,
            "document_workdir_context": BODY_WORKDIR_CONTEXT_RE.pattern,
            "document_software_acronym_context": BODY_SOFTWARE_ACRONYM_CONTEXT_RE.pattern,
            "document_stack_context": BODY_STACK_CONTEXT_RE.pattern,
            "document_system_context": BODY_SYSTEM_CONTEXT_RE.pattern,
            "document_deployment_context": BODY_DEPLOYMENT_RE.pattern,
            "document_source_context": BODY_SOURCE_CONTEXT_RE.pattern,
            "document_page_exclusions": BODY_PAGE_RE.pattern,
            "document_consumer_software": BODY_CONSUMER_SOFTWARE_RE.pattern,
            "document_aggregate_navigation": BODY_AGGREGATE_NAV_RE.pattern,
            "document_browser_consumer": BODY_BROWSER_CONSUMER_RE.pattern,
            "document_navigation_without_strong_context": BODY_NAV_RE.pattern,
            "document_code_marker_diagnostic_only": CODE_MARKER_RE.pattern,
        },
        "gold_rule": "source grade >=2 is positive; grade 0/1 stays in judgments.jsonl and never enters binary qrels",
        "query_scope_revision": {
            "versioned_tools": VERSIONED_QUERY_RE.pattern,
            "development_tools": DEVELOPMENT_TOOL_QUERY_RE.pattern,
            "programming_concepts": PROGRAMMING_CONCEPT_QUERY_RE.pattern,
            "explanation_or_diagnosis": QUERY_EXPLANATION_RE.pattern,
            "toy_program_exclusion": TOY_PROGRAM_QUERY_RE.pattern,
            "nontechnical_query_exclusion": NONTECH_QUERY_RE.pattern,
        },
        "positive_rule": "if any distinct source grade >=2 document is missing or rejected after body filtering, exclude the whole query",
        "dedup_rule": "exact normalized body aliases collapse to one canonical document; source_aliases retains every original ID and all qrels map through that canonical ID",
        "group_rule": "shared canonical positive links, normalized query, and numeric template form connected query groups; IDs alone never create a group",
    }


def _parse_receipt(source_files: list[dict[str, Any]], query_parse: Mapping[str, Any], qrel_parse: Mapping[str, Any], collection_stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "t2ranking-software-zh-receipt-v1",
        "source_files": source_files,
        "queries": query_parse,
        "qrels": qrel_parse,
        "collection": dict(collection_stats),
        "parse_rate_definition": "parsed nonblank UTF-8/TSV rows divided by nonblank rows; malformed rows are never silently promoted",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev", "train"], choices=("dev", "train"), help="original T2Ranking split(s); default: dev train")
    parser.add_argument("--workers", type=int, default=1, help="document-filter worker processes; default: 1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = prepare_release(
            args.raw_root,
            args.output,
            args.evidence_dir,
            splits=tuple(args.splits),
            workers=args.workers,
        )
    except PreparationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "published", "output": str(args.output), "manifest": manifest}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
