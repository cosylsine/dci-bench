#!/usr/bin/env python3
"""Derive a high-precision Chinese software release from a frozen T2Ranking release.

This is a second-layer, CPU-only refinement of the immutable v5 bundle.  It
does not read the 3.6GB source collection and it never invents labels: source
grade>=2 qrels are preserved only when every original positive document passes
the body classifier.  Rows kept in the child release are copied byte-for-byte
at the JSON value level; classification evidence lives outside the release.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import (  # noqa: E402
    file_digest,
    load_bundle,
    publish_bundle,
    verify_release,
)
from dci_bench.data.benchmark_v2 import sample_query_splits  # noqa: E402


FILTER_VERSION = "software-zh-high-precision-v7-refinement"
PARENT_FILTER_VERSION = "software-zh-high-precision-v5"
POLICY = {"max_positive_per_query": 200, "drop_near_duplicates": False}
MIN_DOCUMENTS = 1000
MIN_QUERIES = 60

SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CONTENT_RE = re.compile(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# These patterns intentionally count semantic families, rather than treating
# a single occurrence of "network", "server", "domain", or "installation"
# as a software admission criterion.
STACK_RE = _rx(
    r"(?<![A-Za-z0-9])(?:Python|Java(?:EE|SE|语言)?|JavaScript|C\+\+|C#|C语言|Go语言|Rust|PHP|Ruby|Kotlin|Swift|MATLAB|R语言|\.NET|Node(?:\.js)?|HTML|CSS|SQL|"
    r"MySQL|PostgreSQL|MongoDB|Redis|Linux|Unix|CentOS|Ubuntu|Git|Docker|Kubernetes|Nginx|Apache|Maven|Gradle|Spring(?:Boot|MVC)?|Django|Flask|React|Vue|Qt|"
    r"PowerShell|Visual Studio|VS\s*Code|Eclipse|Android\s*Studio|Android\s*SDK|iOS\s*SDK|FastCGI|IIS|JDBC|JSP|Servlet|XML|JSON|API|SDK|IDE|Shell|Bash|zsh|gcc|GCC|CMake|Makefile)(?![A-Za-z0-9])"
)
CODE_RE = _rx(
    r"代码|源码|程序|函数|变量|线程|指针|语法|脚本|编译|调试|API|SDK|接口|查询|SQL|CREATE\s+TABLE|SELECT\s+.*FROM|命令行|终端|配置文件|部署|依赖|框架|模块|正则|构造函数|析构函数|多线程|匿名函数|闭包|lambda|表达式|算法|数据结构|系统调用|进程|插件|包管理|仓库|分支|import|include|package|public|main方法|注释|注解|数组|数据类型|(?<![A-Za-z])for(?![A-Za-z])|(?<![A-Za-z])while(?![A-Za-z])|break|continue|LIKE|特殊字符|时间戳"
)
DESIGN_RE = _rx(
    r"架构|原理|机制|设计|模型|协议|数据结构|类|对象|模块|组件|服务|引擎|框架|数据库|系统调用|进程|线程|网络|服务器|客户端|前端|后端|接口|配置|部署|文件系统|内核|虚拟机|容器|插件|运行时|编译器|解释器|函数"
)
OP_RE = _rx(
    r"创建|配置|安装|部署|编译|运行|调试|开发|实现|调用|查询|索引|表结构|字段|脚本|命令|终端|日志|排障|依赖|版本|参数|语法|代码|源码|测试|构建|链接|模块|数据库|网络|注册表(?!格)|权限|启动|执行|导入|导出|读写|解析|加载|保存|更新|提交|合并|分支|事务|缓存|添加|编辑|修改|屏蔽|打开|查看|删除|安装语言包"
)
CONFIG_RE = _rx(
    r"配置文件|注册表(?!格)|命令行|终端|脚本|PowerShell|防火墙|系统服务|服务配置|服务器|日志|进程|权限|hosts文件|文件系统|内核|网络协议|TCP|UDP|HTTP|DNS|端口|部署|虚拟机|容器|运行环境"
)
SOFTWARE_CONTEXT_RE = _rx(
    r"软件开发|软件工程|程序设计|编程开发|开发环境|开发者|源代码|源码|数据库|数据结构|命令行|Shell脚本|API接口|前端|后端|全栈|版本控制|开源软件|包管理|Web开发|应用开发|接口调用|服务端|客户端|进程|端口|日志|权限管理|安装包|系统调用|配置文件|注册表(?!格)|部署|依赖|框架|模块|插件|运行时|编译器|解释器|函数式|面向对象|面向过程|多线程|异常处理|指针|正则表达式|Lambda|FastCGI|主键|索引|事务|SQL"
)
INTRINSIC_RE = _rx(
    r"构造函数|析构函数|指针|函数指针|函数|变量|对象|类|线程|进程|系统调用|正则表达式|闭包|匿名函数|lambda|运算符|数据类型|表达式|浅拷贝|深拷贝|多态|继承|接口|API|SDK|IDE|bin\(\)|log\(\)|errno|FastCGI|JDBC|Servlet"
)

# Structural vetoes are evaluated before positive technical signals.  A
# strong word in a university directory or a product listing must not rescue
# an obviously non-software page.
GAME_RE = _rx(
    r"Minecraft|我的世界|游戏名称|游戏类型|游戏制作|游戏发行|游戏特色|游戏攻略|游戏专题|游戏目录|网游|手游|页游|棋牌|连连看|魔兽|DNF|CFHD|吃鸡|王者荣耀|贪吃蛇|九游|玩家|怪物|方块|装备|副本|关卡|道具|预约"
)
GAME_WIKI_RE = _rx(
    r"命名空间ID|红石元件|(?:Java版|基岩版).{0,80}(?:方块|生物|门|音效)|(?:方块|生物).{0,80}(?:Java版|基岩版)"
)
GAME_WORD_RE = _rx(r"游戏")
HARD_EXAM_RE = _rx(
    r"选择题|单选题|多选题|判断题|参考答案|答案查询|题库问题|网课答案|课程完整答案|试题答案|考试题|试卷|题目内容|正确答案|答案列表|题库|答案[:：]"
)
CATALOG_RE = _rx(
    r"学校概况|机构设置|学科建设|专业设置|目录一、|大学年鉴|院系设置|招生简章|学位授权|课程表|图书目录|章节目录|文章列表|相关推荐|最新问题|热门查询"
)
ACADEMIC_RE = _rx(r"硕士论文|博士论文|学位论文|期刊论文|文献综述|参考文献|学报|毕业论文|大学|学院|学校")
MARKET_RE = _rx(
    r"亚马逊|联系我|赚钱|副业|货源|摆地摊|加盟|促销|优惠券|价格|报价|购买|预约|专题站|推荐.{0,12}(?:软件|电脑|产品)|(?:软件|产品).{0,12}(?:推荐|价格|购买|优点|缺点)|网友评分|客服|扫码|二维码|公众号|答案查询网"
)
NONSOFT_RE = _rx(
    r"刑事诉讼|刑事案件|民事诉讼|法律法规|法学|心理学|生物学|化学|医学|药理|护理|疾病|症状|消毒剂|农业|地质|机械加工|磨损|振幅|铣床|车床|加工中心|电机|电路|水产|生态学|汉语语法|诗词|文学|音乐|绘画|哲学|经济学|金融|股票|基金|保险|营销|管理学|行政|建筑|土木|会计|税务|物理学"
)
HARDWARE_RE = _rx(
    r"移动硬盘|硬盘|内存|显卡|主板|网络适配器|网卡|打印机|播放机|显示器|电脑配置|游戏本|轻薄本|安全模式|重装系统|一键重装|装机|默认浏览器|SVCHOST|DNS服务器|U盘"
)
HARDWARE_DOMINANT_RE = _rx(
    r"电流探头|示波器|马达驱动|LED照明|电路|电子整流|仪器分类|测量仪器|电源|芯片|硬件说明|打印机硬件"
)
# Chinese section numbering ("一、", "二、") is common in genuine
# programming tutorials.  Only repeated Latin option markers are evidence of
# a question-bank surface.
OPTION_RE = re.compile(r"(?:^|[。！？\s])[A-D][、.)]")

# A body can contain many programming words while still being a directory,
# answer aggregator, career/product page, or an unrelated science article.
# These patterns are used as structural vetoes or as front-of-body guards.
AGGREGATE_RE = _rx(
    r"首页.{0,60}(?:标签|登录|导航)|标签列表|导航栏|相关推荐|相似问题|更多问题|回答\s*\d+|题目\s*答案|最佳回答|采纳率|结帖|问题点数|查看更多|点击阅读原文|全套笔记|来自度娘|热门文章|最新文章|审查词汇|敏感词|词汇列表"
)
CAREER_RE = _rx(
    r"工作时的心情|工作会来找你|找不到工作|找工作|就业|招聘|职位|岗位|薪资|职业方向|公司待遇|求职|职业规划|工作机会"
)
NONSOFT_SCIENCE_RE = _rx(
    r"植物|根茎|球茎|种子植物|湿地|液压泵|液压马达|泵的流量|机械效率|扭矩|农业科学|向日葵|花盘|水生植物|化学实验"
)
INDUSTRIAL_CATALOG_RE = _rx(
    r"自动化设备|可编程终端|型号命名规则|编程软件是什么|串口.{0,20}(?:电缆|公头|母头)|分辨率是多少|电池型号|通信电缆|画面程序|上传下载画面"
)
CONSUMER_UI_RE = _rx(
    r"电脑常用快捷键|快捷键大全|控制面板|任务栏|鼠标光标|开机弹出网页|浏览器.{0,20}(?:代理|设置)|网页打不开|网络诊断|普通用户|打印机驱动|重装系统|播放视频|忘记密码|强行设置|移除密码|安全模式|文件扩展名|常用文件扩展名|文档密码|Word软件|破解工具|只读模式|图形界面|startx|用户名.{0,12}密码|开机登录|身份证识别|OCR识别软件|上传.*照片|ADSL|MODEM|拨号网络|宽带欠费|路由器"
)
CAD_RE = _rx(r"\bAutoCAD\b|\bCAD\b|CASS|地物编辑|复合线|绘图命令|图层命令|鼠标选取|拟合复合线|等高线|CAD教程")
LIST_AGGREGATE_RE = _rx(
    r"方法一.{0,240}方法二(?:.{0,240}方法三)?|第一种.{0,240}第二种(?:.{0,240}第三种)?|程序\s*\d+.{0,120}程序\s*\d+"
)
MIXED_SURFACE_RE = _rx(
    r"抖音|开学科普季|青少年.{0,20}(?:直播|科学讲堂)|科学讲堂|机器人学|电机拖动|室内设计|3DMax|职业发展的长远规划|面授班|知网(?:查重|算法)|腾讯社招|面试经验|技术一面|技术二面|AndroidSDK开发包国内下载地址|性能测试服务|云数据库Redis版|恢复归档存储对象"
)
NETWORK_HARDWARE_RE = _rx(
    r"交换机|VLAN|端口划分|端口冗余|核心交换机|接入交换机|网段进行管理|TL-SG|TL-SL"
)
FRONT_TOPIC_RE = _rx(
    r"软件开发|软件工程|程序设计|编程(?:语言|开发)?|源代码|源码|代码实现|数据库(?:系统|设计|查询|表|脚本)?|数据表|SQL|SELECT|UPDATE|CREATE\s+TABLE|操作系统|系统调用|命令行|配置文件|服务器(?:配置|部署)?|服务部署|开发环境|编译器|解释器|框架|API接口|API|数据结构|算法(?:实现|分析)?|正则表达式|版本控制|安装包|部署|驱动开发|public\s+class|public\s+static|interface\s+\w+|class\s+[A-Z]"
)
FRONT_STACK_SUPPORT_RE = _rx(
    r"代码|源码|编程|开发|数据库|配置|命令|脚本|程序|函数|算法|数据结构|编译|调试|运行|部署|服务器|服务|框架|模块|API|SQL|日志|进程|权限|变量|指针|数组|参数|返回值|调用|匹配|表达式|接口|类|对象|继承|多态|实现|拷贝|引用|public|class|interface|package|extends|implements"
)
FRONT_CONFIG_RE = _rx(
    r"注册表(?!格)|PowerShell|命令行|系统服务|服务配置|hosts文件|服务器配置|驱动开发"
)
FRONT_CONFIG_ACTION_RE = _rx(
    r"修改|编辑|配置|设置|部署|日志|权限|启动|服务|脚本|命令|保存|排障|解析"
)
CODE_BLOCK_RE = _rx(
    r"#include|int\s+main\s*\(|public\s+static\s+void\s+main|def\s+[A-Za-z_]|import\s+[A-Za-z_][A-Za-z0-9_.]*"
)
STRONG_CONCEPT_RE = _rx(
    r"构造函数|析构函数|多线程|匿名函数|闭包|函数参数|返回值|变量作用域|静态成员|泛型|正则表达式|数据类型|指针|数组|内存|malloc|进程|系统调用|数据库|SQL|算法|数据结构|浅拷贝|深拷贝|继承|多态|引用"
)
CONCEPT_BEHAVIOR_RE = _rx(
    r"定义|调用|返回|执行|编译|匹配|声明|参数|代码|程序|实现|运行|调试|语法|表达式|变量|函数|赋值|对象|引用|类"
)
WINDOWS_SYSTEM_RE = _rx(r"Windows服务|系统进程|svchost|任务管理器|服务宿主|服务管理")
WINDOWS_SYSTEM_ACTION_RE = _rx(r"进程|服务|日志|权限|注册表(?!格)|启动|停止|宿主|系统运行")


def visible_text(text: str) -> str:
    """Return body text with markup and links removed for domain analysis."""
    value = html.unescape(text or "")
    value = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]*>", " ", value)
    value = re.sub(r"(?:https?|ftp)://[^\s<>]+|www\.[^\s<>]+", " ", value, flags=re.I)
    return SPACE_RE.sub(" ", value).strip()


def _terms(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.casefold() for match in pattern.findall(text)}


def _has(pattern: re.Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _repeated_head(body: str) -> bool:
    """Detect a duplicated/truncated page surface without judging its topic."""
    compact = SPACE_RE.sub(" ", body).strip()
    if len(compact) < 240:
        return False
    # A duplicated export often differs after the first sentence or adds a
    # separator, so use a shorter stable prefix than the body evidence shown to
    # reviewers.  This remains conservative for ordinary long articles.
    return compact.count(compact[:60]) >= 2


def _front_has_software_topic(body: str, stacks: set[str]) -> bool:
    """Require a software subject in the first 600 visible characters."""
    first = body[:600]
    if _has(FRONT_TOPIC_RE, first):
        return True
    if _has(FRONT_CONFIG_RE, first) and _has(FRONT_CONFIG_ACTION_RE, first):
        return True
    if _has(WINDOWS_SYSTEM_RE, first) and _has(WINDOWS_SYSTEM_ACTION_RE, first):
        return True
    if _has(_rx(r"HTML|CSS"), first) and _has(_rx(r"标签|元素|样式|网页|代码|浏览器"), first):
        return True
    if _has(STACK_RE, first) and _has(FRONT_STACK_SUPPORT_RE, first):
        return True
    if _has(STRONG_CONCEPT_RE, first) and _has(CONCEPT_BEHAVIOR_RE, first):
        return True
    return False


def _is_code_aggregate(body: str) -> bool:
    """Reject short pages that concatenate several unrelated code snippets."""
    marker_count = _count(CODE_BLOCK_RE, body)
    if marker_count >= 3 and (
        len(body) < 1200
        or _has(_rx(r"程序\s*\d+|小案例|方法一|方法二|多个代码|经典收藏"), body)
    ):
        return True
    return marker_count >= 2 and _has(_rx(r"方法一|方法二|猜数字小游戏|代码片段|多个代码"), body)


def _technical_exception(text: str, stacks: set[str], codes: set[str], ops: set[str], configs: set[str]) -> bool:
    """Recognize a local software concept without accepting bare vocabulary."""
    if len(stacks) >= 1 and len(codes) >= 2 and len(ops) >= 2:
        return True
    if len(stacks) >= 1 and len(codes) >= 1 and len(_terms(DESIGN_RE, text)) >= 2 and len(ops) >= 2:
        return True
    if len(stacks) >= 1 and len(configs) >= 2 and len(ops) >= 2 and _has(
        _rx(r"服务|进程|服务器|开发|部署|系统|项目|环境|代码|脚本|配置|语言包|文件"), text
    ):
        return True
    if _has(_rx(r"数据库|SQL|系统调用|配置文件|部署|服务器|网络协议|命令行|脚本|版本控制|Git|API|JDBC|FastCGI"), text) and len(codes) >= 1 and len(ops) >= 1:
        return True
    if len(codes) >= 2 and len(ops) >= 1 and _has(INTRINSIC_RE, text):
        return True
    # Clear programming concepts can be self-contained even when a page does
    # not repeat the language name in every paragraph.
    if _has(_rx(r"正则表达式"), text) and len(codes) >= 2 and _has(_rx(r"匹配|模式|字符|语法|分组|替换|查找|表达式"), text):
        return True
    if _has(_rx(r"数据类型|指针|变量|数组|结构体|编译"), text) and _has(_rx(r"C语言|C\+\+|程序|代码|函数|内存模型|声明|运算符"), text) and len(codes) >= 2:
        return True
    if _has(_rx(r"Linux|Unix|CentOS|Ubuntu"), text) and _count(_rx(r"命令|日志|终端|进程|服务|权限|文件|脚本|系统"), text) >= 2 and len(ops) >= 1:
        return True
    if _has(_rx(r"hosts文件"), text) and _has(_rx(r"127\.0\.0\.1|编辑|保存|屏蔽|解析|域名"), text) and len(configs) >= 1:
        return True
    if _has(_rx(r"HTML|CSS"), text) and _has(_rx(r"标签|元素|样式|网页|图片|文件|浏览器"), text) and len(codes) >= 1:
        return True
    if _has(_rx(r"HTML|CSS"), text) and _has(_rx(r"标签|元素|样式|网页|图片|img标签|浏览器"), text):
        return True
    if _has(_rx(r"PHP|Python|MATLAB|Java|C语言|SQL"), text) and _has(INTRINSIC_RE, text) and (len(codes) >= 1 or len(ops) >= 2):
        return True
    if _has(_rx(r"PHP"), text) and _has(_rx(r"format|日期|时间|字符串|参数"), text):
        return True
    if len(codes) >= 2 and _has(_rx(r"注释|注解|C语言|Shell|Bash|break|continue|命令行"), text) and (len(ops) >= 1 or len(stacks) >= 1):
        return True
    if _has(_rx(r"API|bin\(\)|log\(\)|errno|FastCGI|JDBC"), text) and len(codes) >= 1 and len(ops) >= 1:
        return True
    return False


def classify_document(text: str) -> dict[str, Any]:
    """Classify one body without consulting query text, grade, or metadata."""
    raw = text or ""
    body = visible_text(raw)
    han_count = len(HAN_RE.findall(body))
    content_chars = len(CONTENT_RE.findall(body))
    chinese_ratio = han_count / max(content_chars, 1)
    stacks = _terms(STACK_RE, body)
    codes = _terms(CODE_RE, body)
    designs = _terms(DESIGN_RE, body)
    ops = _terms(OP_RE, body)
    configs = _terms(CONFIG_RE, body)
    details: dict[str, Any] = {
        "han_count": han_count,
        "content_chars": content_chars,
        "chinese_ratio": round(chinese_ratio, 6),
        "stack_terms": sorted(stacks),
        "code_terms": sorted(codes),
        "design_terms": sorted(designs),
        "operation_terms": sorted(ops),
        "config_terms": sorted(configs),
        "visible_preview": body[:240],
    }

    def result(passed: bool, reason: str) -> dict[str, Any]:
        return {"pass": passed, "reason": reason, **details}

    if han_count < 40 or content_chars < 80 or chinese_ratio < 0.35:
        return result(False, "language")

    first_window = body[:600]
    # Search-result fragments repeatedly elide their text. Privacy notices
    # and game-automation product pages are not software knowledge articles.
    if len(re.findall(r"\.{3,}", first_window)) >= 3:
        return result(False, "truncated_aggregate")
    if _has(_rx(r"隐私权声明|挂机|刷代币|充值卡号|可获得免费调用次数"), first_window):
        return result(False, "nontechnical_surface")
    if _has(CAD_RE, first_window) and not _has(_rx(r"AutoLISP|LISP|二次开发|API|编程|代码实现"), first_window):
        return result(False, "cad_user_operation")
    if body.startswith("考虑因素: 这个工具"):
        return result(False, "missing_subject_context")
    if _has(_rx(r"广义积分|定积分|曲边梯形|积分区间"), body[:200]):
        return result(False, "mathematics_lecture")
    if _has(_rx(r"word(?:文档)?(?:转换|怎么转|转成)|word转换成excel|word怎么转excel"), first_window):
        return result(False, "office_user_operation")
    if _has(_rx(r"一站式.{0,20}(?:解决方案|服务)|为.{0,12}提供专业的"), first_window):
        return result(False, "marketing")
    # Navigation, answer-board, and concatenated-code surfaces are not made
    # relevant by incidental Python/Java/C words. These checks deliberately
    # run before all technical exceptions.
    if _repeated_head(body) or _is_code_aggregate(body) or _has(AGGREGATE_RE, first_window) or _has(LIST_AGGREGATE_RE, first_window):
        return result(False, "aggregate")
    if _has(MIXED_SURFACE_RE, body):
        return result(False, "mixed_surface")

    # A career or job-advice page can mention Linux and code while remaining a
    # social/occupational page. Admit it only when the front contains a
    # concrete implementation/configuration subject.
    if _has(CAREER_RE, first_window) and not _has(
        _rx(r"代码示例|代码实现|配置步骤|安装部署|编译运行|函数调用|数据库查询|命令行参数|开发教程|编程实现"),
        first_window,
    ):
        return result(False, "career")

    front_topic = _front_has_software_topic(body, stacks)
    explicit_front_topic = _has(FRONT_TOPIC_RE, first_window) or (
        _has(FRONT_CONFIG_RE, first_window) and _has(FRONT_CONFIG_ACTION_RE, first_window)
    )
    front_implementation = _has(
        _rx(r"代码实现|源代码|源码|API|数据库|SQL|配置文件|命令行|脚本|系统调用|编译器|程序设计|编程开发|public\s+class"),
        first_window,
    )
    if _has(NONSOFT_SCIENCE_RE, first_window) and not explicit_front_topic:
        return result(False, "science")
    if _has(INDUSTRIAL_CATALOG_RE, first_window) and not front_implementation:
        return result(False, "industrial")
    if _has(CAD_RE, first_window) and not front_implementation:
        return result(False, "cad")
    if _has(CONSUMER_UI_RE, first_window) and not front_implementation:
        return result(False, "consumer_ui")
    if _has(NETWORK_HARDWARE_RE, first_window) and not front_implementation:
        return result(False, "network_hardware")
    if not front_topic:
        return result(False, "no_front_software")

    # These pages remain rejected even if a stray Java/SQL term appears.  A
    # real software development article with a game example can pass only when
    # it has an explicit development topic and several local technical cues.
    game_software_exception = (
        not _has(GAME_WIKI_RE, body)
        and len(stacks) >= 1 and len(codes) >= 2 and len(ops) >= 2
        and _has(_rx(r"数据库|云数据库|集群|备份|Git|版本控制|服务端|服务器|程序|代码|开发"), body)
    )
    if (_has(GAME_RE, body) or _has(GAME_WIKI_RE, body) or (_has(GAME_WORD_RE, body) and _count(GAME_WORD_RE, body) >= 2)) and not game_software_exception:
        game_development = _has(_rx(r"游戏开发|游戏编程|游戏引擎|Unity|Unreal|开发游戏|游戏代码"), body)
        if not (game_development and len(stacks) >= 1 and len(codes) >= 2 and len(ops) >= 2):
            return result(False, "game")

    # Explicit answer/test surfaces are hard rejects.  Generic tutorial words
    # such as "下列" are not enough; option-shaped pages with no explicit
    # answer token are rejected only when they have several option markers.
    if _has(HARD_EXAM_RE, body) or _count(OPTION_RE, body) >= 3:
        return result(False, "exam")

    if _has(CATALOG_RE, body) and not (len(stacks) >= 1 and len(codes) >= 3 and len(ops) >= 3 and _has(_rx(r"数据库|SQL|程序|代码|系统|配置|开发|软件"), body)):
        return result(False, "catalog")

    if _has(MARKET_RE, body) and not (len(stacks) >= 1 and len(codes) >= 3 and len(ops) >= 3 and not _has(GAME_RE, body)):
        return result(False, "marketing")

    software_exception = (
        len(stacks) >= 1 and len(codes) >= 3 and len(ops) >= 3 and _has(
            _rx(r"软件开发|源码|代码实现|系统开发|数据库系统|算法实现|程序实现|API|框架|编译|进程|系统调用|配置|部署"), body
        )
        or (_has(_rx(r"操作系统|内核|进程|线程|系统调用"), body) and len(codes) >= 2 and len(ops) >= 2 and not _has(_rx(r"原创性声明|导师指导|参考文献|摘要|关键词"), body))
    )
    if _has(ACADEMIC_RE, body) and not software_exception:
        return result(False, "academic")

    nonsoftware_exception = (
        (len(stacks) >= 1 and len(codes) >= 2 and len(ops) >= 2 and _has(_rx(r"软件|代码|程序|系统开发|数据库|接口|算法|编程|配置|日志|命令"), body))
        or (_has(_rx(r"MATLAB|Python|Java|C语言|C\+\+"), body) and len(codes) >= 1 and len(ops) >= 1)
        or (len(stacks) >= 1 and len(configs) >= 2 and len(ops) >= 2 and _has(_rx(r"软件|系统|服务|服务器|配置|部署|日志|进程|权限"), body))
    )
    if _has(NONSOFT_RE, body) and not nonsoftware_exception:
        return result(False, "nonsoftware")

    dominant_software = _has(_rx(r"软件开发|源代码|源码|代码|程序|数据库|API|脚本|配置文件|MATLAB|Java|Python|C语言|C\+\+|Spring|Linux|SQL"), first_window)
    matlab_engineering = _has(_rx(r"MATLAB"), first_window) and _has(_rx(r"传递函数|波特图|算法|实现|绘制|仿真"), first_window) and len(ops) >= 2
    if _has(HARDWARE_DOMINANT_RE, first_window) and not ((dominant_software and len(codes) >= 2 and (len(stacks) >= 1 or len(configs) >= 2)) or matlab_engineering):
        return result(False, "hardware_dominant")

    os_concept_exception = _has(_rx(r"操作系统|进程|线程|系统调用"), body) and len(codes) >= 3 and len(ops) >= 3 and _has(_rx(r"程序|代码|系统|内存|执行"), body)
    hardware_topic_exception = (
        front_topic
        and len(codes) >= 2
        and len(ops) >= 1
        and bool(_terms(STACK_RE, body))
        and _has(_rx(r"数据类型|内存|变量|指针|数组|函数|字符串|进程|服务|任务管理器"), body)
    )
    windows_system_exception = (
        _has(WINDOWS_SYSTEM_RE, first_window)
        and _has(WINDOWS_SYSTEM_ACTION_RE, body)
        and _has(_rx(r"Windows|系统|服务|进程"), body)
    )
    hardware_exception = (
        (len(codes) >= 3 and len(ops) >= 3 and (len(stacks) >= 1 or len(configs) >= 2) and _has(_rx(r"软件开发|源码|代码|程序|配置文件|命令行|脚本|注册表(?!格)|驱动开发|系统调用|数据库|服务器|数据类型|变量|指针|数组|结构体|编译"), body))
        or (len(stacks) >= 1 and _has(_rx(r"数据库|数据结构|系统调用|程序|代码|源码"), body) and len(designs) >= 2 and len(ops) >= 2)
        or os_concept_exception
        or hardware_topic_exception
        or windows_system_exception
    )
    if _has(HARDWARE_RE, body) and not hardware_exception:
        return result(False, "hardware_user")

    explicit_anchor = bool(
        stacks
        or _has(SOFTWARE_CONTEXT_RE, body)
        or len(configs) >= 2
        or (len(codes) >= 2 and _has(INTRINSIC_RE, body))
        or (_has(_rx(r"hosts文件"), body) and _has(_rx(r"127\.0\.0\.1|域名解析"), body))
        or windows_system_exception
    )
    if not explicit_anchor:
        return result(False, "no_anchor")
    if windows_system_exception:
        return result(True, "software")
    if _technical_exception(body, stacks, codes, ops, configs):
        return result(True, "software")
    return result(False, "weak")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_positive_ids(qrels: Mapping[str, Mapping[str, Any]], query_id: str) -> list[str]:
    return sorted(str(doc_id) for doc_id, score in qrels.get(query_id, {}).items() if float(score) > 0)


def _select_rows(bundle: Mapping[str, Any]) -> tuple[list[dict], list[dict], dict, list[dict], dict[str, dict], dict[str, dict]]:
    """Classify documents and remove dependent queries without shortening gold."""
    documents: list[dict] = []
    decisions: list[dict] = []
    by_id: dict[str, dict] = {}
    for source_line, row in enumerate(bundle["documents"], 1):
        document_id = row["_id"]
        decision = classify_document(row["text"])
        evidence = {
            "source_line": source_line,
            "doc_id": document_id,
            "decision": "retain" if decision["pass"] else "exclude",
            "reason": decision["reason"],
            "text_sha256": _sha_text(row["text"]),
            "classification": {key: value for key, value in decision.items() if key != "visible_preview"},
        }
        decisions.append(evidence)
        by_id[document_id] = decision
        if decision["pass"]:
            documents.append(row)

    retained_documents = {row["_id"] for row in documents}
    queries: list[dict] = []
    query_decisions: list[dict] = []
    kept_queries: dict[str, dict] = {}
    for source_line, row in enumerate(bundle["queries"], 1):
        query_id = row["_id"]
        positive_ids = _source_positive_ids(bundle["qrels"], query_id)
        rejected = sorted(doc_id for doc_id in positive_ids if doc_id not in retained_documents)
        kept = bool(positive_ids) and not rejected
        query_decisions.append({
            "source_line": source_line,
            "query_id": query_id,
            "decision": "retain" if kept else "exclude",
            "reason": "all_original_positive_documents_retained" if kept else "positive_document_rejected",
            "split": (row.get("metadata") or {}).get("original_split"),
            "text_sha256": _sha_text(row["text"]),
            "original_positive_ids": positive_ids,
            "rejected_positive_ids": rejected,
        })
        if kept:
            queries.append(row)
            kept_queries[query_id] = row

    kept_qrels = {query_id: copy.deepcopy(bundle["qrels"][query_id]) for query_id in kept_queries}
    kept_ids = set(kept_queries)
    kept_judgments = [
        row for row in bundle.get("judgments", [])
        if row.get("query_id") in kept_ids and row.get("doc_id") in retained_documents
    ]
    kept_evidence = [
        row for row in bundle.get("evidence", [])
        if row.get("query_id") in kept_ids and row.get("doc_id") in retained_documents
    ]
    return documents, queries, kept_qrels, kept_judgments, by_id, {
        "document_decisions": decisions,
        "query_decisions": query_decisions,
        "evidence": kept_evidence,
    }


def _audit_sample(bundle: Mapping[str, Any], decisions_by_id: Mapping[str, dict]) -> list[dict]:
    rows = sorted(bundle["documents"], key=lambda row: hashlib.sha256(row["_id"].encode("utf-8")).hexdigest())[:50]
    notes = {
        "game": "游戏/wiki/目录内容，游戏或方块语义主导正文",
        "catalog": "学校/目录/导航聚合结构，软件词不足以改变主题",
        "marketing": "商品、营销或推荐页面，缺少开发语义",
        "exam": "考试、题库、答案或选项聚合页面",
        "academic": "论文/学术声明页面，缺少软件实现主体",
        "nonsoftware": "法律、心理、工程等非软件主题占主导",
        "hardware_dominant": "硬件/测量主题在正文前部占主导",
        "hardware_user": "消费硬件或用户操作页面，缺少开发配置语义",
        "aggregate": "正文前部或结构显示导航/问答/重复/拼接片段，主题不构成单一技术文档",
        "career": "职业、招聘或工作建议页面，缺少具体实现或配置主体",
        "science": "非软件科学或农业主题在正文前部占主导",
        "industrial": "工业设备/型号/电缆目录或操作说明，缺少软件开发主体",
        "cad": "CAD/测绘绘图操作说明，缺少软件开发或系统配置主体",
        "consumer_ui": "消费用户界面或普通故障操作页，缺少开发配置主体",
        "mixed_surface": "软件词与教育、营销、论文、职业或产品目录内容混杂，前后不构成单一技术文档",
        "network_hardware": "交换机/VLAN/端口规划等网络硬件方案，缺少软件实现或系统配置主体",
        "no_front_software": "正文前600字没有独立的软件开发、配置、系统、数据库或技术栈主题",
        "weak": "存在软件词但缺少独立开发/配置/系统语义",
        "no_anchor": "没有明确软件主题锚点",
        "language": "正文过短或中文占比不足",
        "software": "正文含明确软件栈及多项开发、配置、系统或数据库语义",
        "truncated_aggregate": "正文前部重复出现截断省略号，缺少连续完整的技术叙述",
        "nontechnical_surface": "隐私声明或游戏挂机等非技术知识页面",
        "cad_user_operation": "CAD普通界面操作，未提供开发或扩展接口知识",
        "missing_subject_context": "依赖未命名的上文工具，缺少独立主体",
        "mathematics_lecture": "数学课程主体，仅附带提及软件试验",
        "office_user_operation": "办公文档格式转换的普通界面操作",
    }
    return [
        {
            "rank": rank,
            "doc_id": row["_id"],
            "decision": "retain" if decisions_by_id[row["_id"]]["pass"] else "exclude",
            "reason": decisions_by_id[row["_id"]]["reason"],
            "text_sha256": _sha_text(row["text"]),
            "preview": visible_text(row["text"])[:600],
            "review_note": notes[decisions_by_id[row["_id"]]["reason"]],
        }
        for rank, row in enumerate(rows, 1)
    ]


V7_RETAINED_SAMPLE_REVIEW_NOTES: dict[str, str] = {
    "706221": "Java SecurityManager权限检查、异常和调用接口说明。",
    "1264166": "异或交换的性质与C代码解释，首段有完整技术内容。",
    "1305792": "Linux more命令功能、参数和操作键说明。",
    "2213601": "C/C++ strcpy指针检查、NULL与类型转换解释。",
    "1454172": "Jenkins node块调度和工作区语义。",
    "408407": "Grafana服务端渲染、配置和故障日志。",
    "516757": "Windows DPI兼容问题及注册表排障，不仅是界面截图。",
    "2172224": "组策略和注册表修复任务管理器权限。",
    "2070258": "Linux zip/unzip/tar命令及参数。",
    "1767191": "Python语言设计原则与语法取舍。",
    "1318203": "DMI管理接口与SNMP映射；存在少量文本噪声，核心定义完整。",
    "537861": "JDK环境变量、profile加载和版本验证。",
    "2242608": "C运算符优先级与结合性解释。",
    "653907": "防火墙入站/出站规则及MySQL端口示例；有重复段落但语义独立。",
    "1265691": "恶意软件行为、注册表与进程特征描述，属于软件安全知识。",
    "1325986": "Maven依赖管理及环境变量配置，首段有独立含义。",
    "382125": "regexp_substr参数和分割用法；目录摘要后有完整参数释义。",
    "7075": "C#窗体、DataGridView及数据库连接教学正文。",
    "1317957": "SQL decode条件与返回值、比较大小示例；首段完整。",
    "268121": "Linux login和startx启动图形环境的命令说明。",
    "2231864": "HTML font/size及标题标签字体规则，关键关系在正文中。",
    "1031600": "C++常量定义、数值表示与使用示例。",
    "1326003": "Azure MySQL Workbench导入导出条件与步骤。",
    "647890": "Facebook技术架构、LAMP和缓存；开头产品介绍后有独立技术主体。",
    "598909": "静态网页和动态网页的技术差异。",
    "227375": "CSS/HTML框架用途和特征介绍。",
    "2183041": "编译地址、运行地址与bootloader解释。",
    "222978": "Linux共享内存及mmap机制。",
    "1189960": "小程序WXS事件与模块导出。",
    "487441": "Shell脚本日志和日期变量示例。",
    "2125507": "数据库管理系统定义及分类。",
    "1433092": "默认路由、网关、子网及route print解释。",
    "1303314": "IntelliJ IDEA跨文件搜索及正则选项。",
    "1318503": "DOM接口测试和动态修改页面的说明。",
    "1283770": "SSH协议、安装命令及使用。",
    "2208535": "SQL连接查询及嵌套循环连接机制。",
    "1679927": "Notepad++正则邮箱匹配，表达式和操作在文本中。",
    "625425": "恶意软件家族、实现语言和行为特征分类。",
    "267004": "Java方法重载概念与实例。",
    "650025": "Java classpath配置及加载顺序。",
    "545090": "大数据处理的技术特征和数据架构背景。",
    "312235": "文本检索软件的检索、压缩包和编码支持；首段技术描述独立。",
    "1293016": "Python基础数据类型。",
    "2281952": "JSP、Servlet、MVC和Java代码组织。",
    "822840": "URL语法、绝对与相对路径。",
    "1410486": "Python/TensorFlow/OpenCV车牌识别实现。",
    "1321979": "HTML结构教学；原文保留标记，正文解释标签作用。",
    "1766607": "框架与设计模式差异及Web框架示例。",
    "794361": "PLC/DCS功能块编程、触发器和程序逻辑；包含独立的软件编程定义。",
    "1283654": "Spring Boot profile、日志配置和启动参数，核心配置方法在文本中。"
}


def _audit_retained_sample(documents: Iterable[Mapping[str, Any]], decisions_by_id: Mapping[str, dict]) -> list[dict]:
    """Return the reviewed child sample using a domain-independent hash order."""
    rows = sorted(
        documents,
        key=lambda row: hashlib.sha256(("root-audit:" + row["_id"]).encode("utf-8")).hexdigest(),
    )[:50]
    if len(rows) < 50:
        raise ValueError(f"retained sample requires 50 documents, got {len(rows)}")
    missing = [row["_id"] for row in rows if row["_id"] not in V7_RETAINED_SAMPLE_REVIEW_NOTES]
    if missing:
        raise RuntimeError(f"retained sample lacks manual review notes for {missing}")
    return [
        {
            "rank": rank,
            "doc_id": row["_id"],
            "sample_key_sha256": hashlib.sha256(("root-audit:" + row["_id"]).encode("utf-8")).hexdigest(),
            "decision": "retain",
            "reason": decisions_by_id[row["_id"]]["reason"],
            "text_sha256": _sha_text(row["text"]),
            "preview_first_600": visible_text(row["text"])[:600],
            "review_context_first_1200": visible_text(row["text"])[:1200],
            "review_note": V7_RETAINED_SAMPLE_REVIEW_NOTES[row["_id"]],
        }
        for rank, row in enumerate(rows, 1)
    ]


def _split_precheck(child: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    # The child split graph is regenerated by publish_bundle; read it later in
    # main so that this function remains usable with an in-memory test bundle.
    query_ids = {row["_id"] for row in child["queries"]}
    source_counts = Counter((row.get("metadata") or {}).get("original_split", "unknown") for row in child["queries"])
    sampling_records = [
        {
            "query_id": row["_id"],
            "group_id": (row.get("metadata") or {}).get("group_id", f"query-group-{row['_id']}"),
        }
        for row in child["queries"]
    ]
    benchmark_sampling = sample_query_splits(
        sampling_records,
        dev_size=10,
        test_size=50,
        smoke_size=1,
        seed=42,
        task_id="software_zh",
        group_key="group_id",
    )
    if len(benchmark_sampling["dev"]) != 10 or len(benchmark_sampling["test"]) != 50 or len(benchmark_sampling["smoke"]) != 1:
        raise AssertionError(f"unexpected benchmark sample sizes: {benchmark_sampling}")
    if set(benchmark_sampling["dev"]) & set(benchmark_sampling["test"]):
        raise AssertionError("benchmark dev/test sample overlap")
    if not set(benchmark_sampling["smoke"]).issubset(set(benchmark_sampling["dev"])):
        raise AssertionError("benchmark smoke sample is not a dev subset")
    return {
        "query_count": len(query_ids),
        "query_ids_unique": len(query_ids) == len(child["queries"]),
        "original_split_counts": dict(sorted(source_counts.items())),
        "qrels_query_coverage": set(child["qrels"]) == query_ids,
        "qrel_document_ids_present": all(
            doc_id in {row["_id"] for row in child["documents"]}
            for scores in child["qrels"].values() for doc_id in scores
        ),
        "parent_query_count": len(parent["queries"]),
        "parent_positive_pairs": sum(len(row) for row in parent["qrels"].values()),
        "benchmark_sampling": {
            "seed": 42,
            "group_key": "group_id",
            "dev_size": 10,
            "test_size": 50,
            "smoke_size": 1,
            "task_id": "software_zh",
            "splits": benchmark_sampling,
        },
    }


def refine(parent_release: Path, output: Path, evidence_dir: Path, *, min_documents: int = MIN_DOCUMENTS, min_queries: int = MIN_QUERIES) -> dict[str, Any]:
    """Create a new immutable v7 child release and complete evidence."""
    parent_release = Path(parent_release)
    output = Path(output)
    evidence_dir = Path(evidence_dir)
    if output.exists() or evidence_dir.exists():
        raise FileExistsError("refinement output/evidence must be new directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    parent_manifest = verify_release(parent_release)
    parent = load_bundle(parent_release)
    if parent_manifest.get("state") != "labels_ready":
        raise ValueError(f"parent release is not labels_ready: {parent_manifest.get('state')}")
    parent_filter = ((parent.get("provenance") or {}).get("selection_policy") or {}).get("filter_version")
    if parent_filter != PARENT_FILTER_VERSION:
        raise ValueError(f"unexpected parent filter version: {parent_filter!r}")

    documents, queries, qrels, judgments, decisions_by_id, aux = _select_rows(parent)
    reason_counts = Counter(item["reason"] for item in aux["document_decisions"])
    query_reason_counts = Counter(item["reason"] for item in aux["query_decisions"])
    positive_ids = {doc_id for row in parent["qrels"].values() for doc_id, score in row.items() if float(score) > 0}
    positive_decisions = [item for item in aux["document_decisions"] if item["doc_id"] in positive_ids]
    positive_pass = sum(item["decision"] == "retain" for item in positive_decisions)
    retained_query_ids = {row["_id"] for row in queries}
    if len(documents) < min_documents or len(queries) < min_queries:
        raise RuntimeError(f"refinement thresholds failed: documents={len(documents)}, queries={len(queries)}")
    if any(doc_id not in {row["_id"] for row in documents} for query_id in retained_query_ids for doc_id in qrels[query_id]):
        raise AssertionError("retained qrels refer to a filtered document")

    child = copy.deepcopy(parent)
    child["documents"] = documents
    child["queries"] = queries
    child["qrels"] = qrels
    child["judgments"] = judgments
    child["evidence"] = aux["evidence"]
    provenance = child.setdefault("provenance", {})
    provenance["selection_policy"] = copy.deepcopy(provenance.get("selection_policy", {}))
    provenance["selection_policy"]["filter_version"] = FILTER_VERSION
    provenance["selection_policy"]["parent_filter_version"] = PARENT_FILTER_VERSION
    provenance["selection_policy"]["second_layer_body_only"] = True
    provenance["selection_policy"]["all_grade_ge_2_positives_required"] = True
    provenance["refinement"] = {
        "filter_version": FILTER_VERSION,
        "parent_release": str(parent_release),
        "parent_completion_sha256": file_digest(parent_release / "_COMPLETE.json"),
        "classifier_input": "document.text only; no query, grade, metadata, or raw collection",
        "source_document_count": len(parent["documents"]),
        "retained_document_count": len(documents),
        "source_query_count": len(parent["queries"]),
        "retained_query_count": len(queries),
        "source_positive_pair_count": sum(len(row) for row in parent["qrels"].values()),
        "retained_positive_pair_count": sum(len(row) for row in qrels.values()),
        "script_sha256": file_digest(Path(__file__)),
    }
    specification = copy.deepcopy(parent_manifest.get("specification") or {})
    specification.setdefault("options", {})["filter_version"] = FILTER_VERSION
    specification["options"]["parent_release"] = str(parent_release)
    specification["options"]["second_layer"] = "body_only_high_precision"
    parent_completion_sha = file_digest(parent_release / "_COMPLETE.json")
    specification["options"]["parent_completion_sha256"] = parent_completion_sha

    # The content review must finish before a release completion marker exists.
    parent_sample = _audit_sample(parent, decisions_by_id)
    retained_sample = _audit_retained_sample(documents, decisions_by_id)
    manifest = publish_bundle(child, output, specification=specification, policy=POLICY, parent_release=parent_release)
    child_published = load_bundle(output)
    verify_release(output)

    # Publish validation: every retained row is an exact source row, and every
    # original positive on every retained query still has its original qrel and
    # judgment rows.  This catches accidental quality-stage rewrites.
    parent_docs = {row["_id"]: row for row in parent["documents"]}
    child_docs = {row["_id"]: row for row in child_published["documents"]}
    parent_queries = {row["_id"]: row for row in parent["queries"]}
    child_queries = {row["_id"]: row for row in child_published["queries"]}
    if any(child_docs[doc_id] != parent_docs[doc_id] for doc_id in child_docs):
        raise AssertionError("published document row changed")
    if any(child_queries[query_id] != parent_queries[query_id] for query_id in child_queries):
        raise AssertionError("published query row changed")
    if child_published["qrels"] != qrels:
        raise AssertionError("published qrels changed")
    parent_judgment_set = {(json.dumps(row, sort_keys=True, ensure_ascii=False)) for row in parent["judgments"]}
    if any(json.dumps(row, sort_keys=True, ensure_ascii=False) not in parent_judgment_set for row in child_published["judgments"]):
        raise AssertionError("published judgment was not present in parent")
    positive_proof = []
    parent_judgments = parent["judgments"]
    child_judgment_set = {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in child_published["judgments"]}
    for query_id in sorted(retained_query_ids):
        ids = _source_positive_ids(parent["qrels"], query_id)
        source_rows = [row for row in parent_judgments if row.get("query_id") == query_id and row.get("doc_id") in ids and float(row.get("original_grade", 0)) >= 2]
        missing_rows = [row for row in source_rows if json.dumps(row, sort_keys=True, ensure_ascii=False) not in child_judgment_set]
        positive_proof.append({
            "query_id": query_id,
            "positive_doc_ids": ids,
            "child_qrel_doc_ids": sorted(child_published["qrels"][query_id]),
            "source_grade_ge_2_rows": len(source_rows),
            "missing_source_positive_judgments": len(missing_rows),
            "preserved": not missing_rows and ids == sorted(child_published["qrels"][query_id]),
        })
    if not all(row["preserved"] for row in positive_proof):
        raise AssertionError("positive preservation proof failed")

    split_precheck = _split_precheck(child_published, parent)
    child_splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    split_precheck.update({
        "child_component_count": child_splits.get("component_count"),
        "child_split_counts": child_splits.get("query_splits", {}).get("counts", child_splits.get("counts", {})),
        "child_split_query_ids": child_splits.get("query_splits", {}).get("queries", {}),
        "cross_original_split_components": sum(
            len({(child_queries[q].get("metadata") or {}).get("original_split") for q in component.get("queries", [])}) > 1
            for component in child_splits.get("components", [])
        ),
    })
    summary = {
        "filter_version": FILTER_VERSION,
        "parent_release": str(parent_release),
        "parent_completion_sha256": parent_completion_sha,
        "script_sha256": file_digest(Path(__file__)),
        "counts": {
            "parent_documents": len(parent["documents"]),
            "retained_documents": len(documents),
            "excluded_documents": len(parent["documents"]) - len(documents),
            "parent_queries": len(parent["queries"]),
            "retained_queries": len(queries),
            "excluded_queries": len(parent["queries"]) - len(queries),
            "parent_positive_pairs": sum(len(row) for row in parent["qrels"].values()),
            "retained_positive_pairs": sum(len(row) for row in qrels.values()),
            "parent_positive_documents": len(positive_ids),
            "retained_positive_documents": positive_pass,
            "retained_judgments": len(judgments),
        },
        "document_reason_counts": dict(sorted(reason_counts.items())),
        "query_reason_counts": dict(sorted(query_reason_counts.items())),
        "positive_decision_counts": dict(sorted(Counter(item["decision"] for item in positive_decisions).items())),
        "split_precheck": split_precheck,
        "retained_sample": {
            "algorithm": "sha256('root-audit:' + doc_id)",
            "count": len(retained_sample),
            "path": "sample_50_hash_audit.jsonl",
            "manual_review_required": True,
        },
        "release_manifest": manifest,
        "boundary": "Second-layer body-only refinement of frozen v5; no raw collection scan and no model evaluation.",
    }
    evidence_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(evidence_dir / "document_decisions.jsonl", aux["document_decisions"])
    _write_jsonl(evidence_dir / "query_decisions.jsonl", aux["query_decisions"])
    _write_jsonl(evidence_dir / "positive_preservation.jsonl", positive_proof)
    _write_jsonl(evidence_dir / "parent_sample_50_hash_audit.jsonl", parent_sample)
    _write_jsonl(evidence_dir / "sample_50_hash_audit.jsonl", retained_sample)
    _write_jsonl(evidence_dir / "rejected_positive_examples.jsonl", [
        item for item in aux["document_decisions"] if item["doc_id"] in positive_ids and item["decision"] == "exclude"
    ])
    _write_json(evidence_dir / "split_precheck.json", split_precheck)
    _write_json(evidence_dir / "benchmark_sampling_preview.json", split_precheck["benchmark_sampling"])
    _write_json(evidence_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--min-documents", type=int, default=MIN_DOCUMENTS)
    parser.add_argument("--min-queries", type=int, default=MIN_QUERIES)
    args = parser.parse_args()
    summary = refine(args.parent_release, args.output, args.evidence_dir, min_documents=args.min_documents, min_queries=args.min_queries)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
