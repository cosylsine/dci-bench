"""Focused contracts for the offline T2Ranking software-zh preparer."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import scripts.prepare_t2ranking_zh_20261102 as preparer
from scripts.prepare_t2ranking_zh_20261102 import (
    PreparationBlocked,
    QueryInfo,
    _body_aliases,
    _expected_small_files,
    _stream_collection,
    classify_document,
    classify_query,
    _preview_query_splits,
    evaluate_queries,
    group_queries,
    prepare_release,
)
from dci_bench.data.pipeline.core import export_dci, load_bundle, verify_release


def test_versioned_tools_and_intrinsic_concepts_keep_their_software_context():
    for query in (
        'centos6安装python3', 'python3.8安装pip', 'centos7修改ssh端口号',
        'mysql5.7安装报错 3304', 'jdk无法打开jar文件', '谈谈对jvm的理解',
        'pycharm怎么运行单行代码', 'jdbc分页查询数据重复',
        'new运算符会调用构造函数吗', '空指针内存为什么不能访问',
        '函数指针与指针函数的区别', '用c语言编写的程序被称为什么',
    ):
        assert classify_query(query)['pass'], query
    for query in (
        '呼吸机pip什么参数', '松下npm网络怎么设置', 'pythonfoo是什么意思',
        'python用递归函数打印斐波那契', 'matlab求函数面积',
        'python输出杨辉三角形', 'python程序员学历要求', '夏普电视机操作系统',
    ):
        assert not classify_query(query)['pass'], query
    assert not classify_document(
        '怀旧大脚插件无法启动，配置文件下载失败，客户端无法启动。'
        'IE选项重置无法解决；取消自动配置脚本仍然无法解决，'
        '删除Cache、WTF、Interface并重新安装还是无法解决，请求帮助。'
    )['pass']


def _query(key: str, source_id: str, text: str, split: str = "dev") -> QueryInfo:
    from scripts.prepare_t2ranking_zh_20261102 import normalize_query, query_template

    return QueryInfo(key, split, source_id, text, normalize_query(text), query_template(text), ("development",))


def _doc(text: str = "Python 软件开发需要清晰的配置、依赖、调试和版本管理说明。") -> dict:
    return {"_id": "unused", "text": text}


def test_any_unusable_grade_ge_2_positive_excludes_the_whole_query() -> None:
    queries = {"dev:q": _query("dev:q", "q", "Python 安装配置")}
    qrels = {"dev:q": [
        {"doc_id": "good", "original_grade": 2, "source_split": "dev", "source_id": "q", "line": 1},
        {"doc_id": "missing", "original_grade": 3, "source_split": "dev", "source_id": "q", "line": 2},
    ]}
    eligible, decisions, _ = evaluate_queries(queries, qrels, {"good": _doc()}, {})
    assert eligible == {}
    assert decisions[0]["status"] == "excluded"
    assert "positive_unusable_whole_query" in decisions[0]["reasons"]


def test_more_than_200_positive_documents_excludes_the_query() -> None:
    queries = {"dev:q": _query("dev:q", "q", "Python 数据库配置")}
    rows = [
        {"doc_id": f"d{i}", "original_grade": 2, "source_split": "dev", "source_id": "q", "line": i}
        for i in range(201)
    ]
    documents = {row["doc_id"]: _doc() for row in rows}
    eligible, decisions, _ = evaluate_queries(queries, {"dev:q": rows}, documents, {}, max_positive=200)
    assert eligible == {}
    assert "positive_count_over_limit" in decisions[0]["reasons"]


def test_exact_body_aliases_keep_each_original_id() -> None:
    body = "Python 软件开发需要清晰的配置、依赖、调试和版本管理说明。"
    docs = {"p1": {"_id": "p1", "text": body}, "p2": {"_id": "p2", "text": "  " + body.replace(" ", "\u3000")}}
    source_to_canonical, canonical_docs, groups = _body_aliases(docs)
    assert list(groups.values()) == [["p1", "p2"]]
    assert source_to_canonical == {"p1": "p1", "p2": "p1"}
    assert list(canonical_docs) == ["p1"]
    assert canonical_docs["p1"]["metadata"]["source_aliases"] == ["p1", "p2"]


def test_only_grade_ge_2_enters_binary_positive_but_all_grades_stay_in_judgments() -> None:
    queries = {"dev:q": _query("dev:q", "q", "Python 安装配置")}
    rows = [
        {"doc_id": "zero", "original_grade": 0, "source_split": "dev", "source_id": "q", "line": 1},
        {"doc_id": "one", "original_grade": 1, "source_split": "dev", "source_id": "q", "line": 2},
        {"doc_id": "two", "original_grade": 2, "source_split": "dev", "source_id": "q", "line": 3},
    ]
    docs = {row["doc_id"]: _doc() for row in rows}
    eligible, _, retention = evaluate_queries(queries, {"dev:q": rows}, docs, {})
    assert eligible["dev:q"]["positive_ids"] == ["two"]
    assert [row["original_grade"] for row in eligible["dev:q"]["judgments"]] == [0, 1, 2]
    assert retention[0]["binary_qrel"] == 1
    assert retention[0]["canonical_doc_id"] == "two"


def test_positive_aliases_map_to_one_canonical_and_each_source_grade_is_proven() -> None:
    body = "Python 软件开发需要清晰的配置、依赖、调试和版本管理说明。"
    source_docs = {"p1": {"_id": "p1", "text": body}, "p2": {"_id": "p2", "text": " " + body}}
    source_to_canonical, canonical_docs, aliases = _body_aliases(source_docs)
    rows = [
        {"doc_id": "p1", "original_grade": 2, "source_split": "dev", "source_id": "q", "line": 1},
        {"doc_id": "p2", "original_grade": 3, "source_split": "dev", "source_id": "q", "line": 2},
        {"doc_id": "missing", "original_grade": 0, "source_split": "dev", "source_id": "q", "line": 3},
    ]
    eligible, _, retention = evaluate_queries(
        {"dev:q": _query("dev:q", "q", "Python 安装配置")},
        {"dev:q": rows},
        canonical_docs,
        {},
        source_to_canonical,
        alias_groups=aliases,
    )
    assert eligible["dev:q"]["positive_ids"] == ["p1"]
    assert [(row["source_doc_id"], row["canonical_doc_id"]) for row in retention] == [("p1", "p1"), ("p2", "p1")]
    assert [row["original_grade"] for row in eligible["dev:q"]["judgments"]] == [2, 3, 0]


def test_seed42_preview_uses_benchmark_v2_sampler_and_group_key() -> None:
    records = [{"query_id": f"q{i:03d}", "group_id": f"g{i:03d}"} for i in range(70)]
    preview = _preview_query_splits(records)
    assert len(preview["dev"]) == 10
    assert len(preview["test"]) == 50
    assert len(preview["smoke"]) == 1
    assert set(preview["dev"]).isdisjoint(preview["test"])


def test_query_filter_rejects_ambiguous_nonsoftware_and_consumer_terms() -> None:
    rejected = (
        "widely后面加什么",
        "digital化方案",
        "黄冈师范学院院校代码",
        "中银香港swift代码是什么",
        "依赖心理",
        "政策框架是什么",
        "不同容器的压力",
        "Android手机刷机",
        "iOS微信视频怎么显示",
        "Windows Store安装位置",
        "网络用语是什么意思",
    )
    for text in rejected:
        assert not classify_query(text)["pass"], text
    accepted = (
        "Python安装配置",
        "Windows系统配置",
        "Android Studio调试",
        "Git拉取项目",
        "Docker容器部署",
        "Linux下Java应用部署",
        "配置防火墙端口",
        "配置文件下载失败客户端无法启动",
    )
    for text in accepted:
        assert classify_query(text)["pass"], text


def test_query_filter_rejects_generic_infrastructure_and_pure_code_tasks() -> None:
    rejected = (
        "网络适配器常用的分类方法",
        "上网设置代理作用",
        "虚拟机开蓝牙",
        "笔记本网络感叹号",
        "服务器异常下载不了软件",
        "Windows安全模式重装",
        "IP地址是什么意思",
        "无线网网络布线",
        "热点或域网络联系管理员",
        "单窗口单IP",
        "域名是什么",
        "宽带DNS",
        "QQ浏览器开发者",
        "colg IP白名单",
        "爱快路由",
        "Win10分区安装",
        "程序下载解析安装包",
        "python统计不同字符的个数",
        "c语言相反次序重新组合",
        "python简单求和",
        "java 输出list内容",
        "反转链表java",
        "python递归例子",
    )
    for text in rejected:
        assert not classify_query(text)["pass"], text
    # A concrete error/debug or function explanation remains eligible when it
    # names a real software context.
    accepted = (
        "javascript求和1加1等于11哪错了",
        "数据库加减乘除函数",
        "c语言不合法标识符",
    )
    for text in accepted:
        assert classify_query(text)["pass"], text


def test_document_filter_applies_mobile_and_windows_context_guards() -> None:
    consumer_windows = "Windows 壁纸、播放器和普通桌面使用说明，不包含开发、配置或系统排障内容。" * 2
    technical_windows = "Windows 系统配置和服务排障需要检查日志、权限、网络与进程状态，并记录服务配置和错误日志。" * 2
    consumer_mobile = "Android 手机刷机、装机和视频播放的消费使用说明，仅供普通用户使用。" * 2
    technical_mobile = "Android SDK 应用开发需要配置调试环境、接口权限和构建工具。" * 3
    assert not classify_document(consumer_windows)["pass"]
    assert classify_document(technical_windows)["pass"]
    assert not classify_document(consumer_mobile)["pass"]
    assert classify_document(technical_mobile)["pass"]


def test_document_filter_rejects_weak_network_consumer_academic_and_navigation_bodies() -> None:
    rejected = (
        "网络监控一根网线就能完成线路，支持手机远程监看，客户端电脑录像并可回放。" * 2,
        "中国联通推出5G套餐和流量包，用户可以购买升级包，比较不同运营商的优惠价格。" * 2,
        "厦门大学学位论文原创性声明，本论文在导师指导下完成，研究成果将进入学校数据库。" * 2,
        "登录评论点赞收藏视频，点击下一页查看答案，下载前需要安装客户端。" * 3,
        "现在重装系统的软件很多，一键重装和备份还原都很方便，普通用户可以按步骤操作。" * 2,
        "安卓手机连接电脑后打开USB调试，进入设置点击开发者选项即可完成操作。" * 2,
    )
    for text in rejected:
        assert not classify_document(text)["pass"], text[:40]
    accepted = (
        "Linting Python 会运行程序并分析代码中的潜在错误，VS Code 可以配置断点、测试和调试环境。" * 3,
        "MySQL 数据库实验创建数据表并插入数据，使用主键、查询语句和索引验证表结构。" * 3,
        "VSS 版本控制连接服务器共享目录后获取源代码项目，并设置工作目录进行签入和签出。" * 3,
        "Linux 系统配置和服务排障需要检查日志、权限与进程状态，并通过命令行完成服务部署。" * 3,
    )
    for text in accepted:
        decision = classify_document(text)
        assert decision["pass"], decision
        assert decision["content_chars"] >= 80
        assert decision["chinese_ratio"] >= 0.35


def test_document_filter_requires_independent_software_semantics_and_rejects_hardware_marketing() -> None:
    acronym_only = (
        "这是一篇普通产品说明，介绍服务体验和日常使用方法。"
        "API只是页面中偶然出现的缩写，不谈技术细节。本文帮助用户登录与使用。"
    ) * 2
    camera_marketing = (
        "高清网络摄像头支持SDK和API接口，连接网络即可观看并录像。"
        "产品支持多用户和云服务，适合家庭和商铺使用。"
    ) * 3
    assert classify_document(acronym_only)["reason"] == "no_explicit_software_context"
    assert classify_document(camera_marketing)["reason"] == "consumer_hardware"

    api_development = (
        "API接口调用需要配置鉴权参数，开发者可在IDE中调试请求并查看返回日志。"
        "本文说明项目集成和错误排查流程。"
    ) * 3
    driver_development = (
        "Linux设备驱动开发需要实现硬件接口和内核模块，使用C语言编写代码，"
        "并通过日志排查设备中断和协议问题。"
    ) * 3
    assert classify_document(api_development)["pass"]
    assert classify_document(driver_development)["pass"]


def test_document_filter_rejects_incidental_software_terms_in_other_domains() -> None:
    rejected = (
        "职位基本要求：本科以上学历，熟悉Java、HTML、JavaScript和Spring框架，"
        "有三年以上工作经验，负责项目开发和面试。" * 2,
        "西安邮电大学录取分数线和各省本科招生信息如下，考生可查询专业和学校排名。"
        "计算机科学与技术、软件工程等专业按批次录取。" * 2,
        "诊所医生介绍患者治疗和手术过程，文章中偶然出现HTML页面标记和软件安装说明。"
        "请咨询医院了解具体情况。" * 2,
        "弹性云服务器三年低至5折，多种配置可选，了解详情并购买产品。"
        "服务热线欢迎咨询，帮助用户选择合适的云服务套餐。" * 2,
        "Office 365离线安装程序需要登录帐户，下载完成后按照页面步骤安装使用。"
        "普通用户可以选择语言和位数。" * 2,
        "打开电脑发现光驱不见了，点击开始菜单并执行命令即可恢复设备。"
        "如果仍然无法使用，请重新安装软件。" * 2,
        "大棚.HTML词典查询页面提供热门查询和相关词语大全，用户可以点击结果继续浏览。" * 2,
        "Windows开机密码查看器是一款免费软件，软件类型为密码破解软件，"
        "应用平台为Windows，用户双击运行即可获取密码。" * 2,
        "PKPM结构设计软件生成工程图元，工作目录中调用AUTOCAD程序修改文字。"
        "本文介绍建筑设计和结构设计中的命令。" * 2,
        "红警Game.exe兼容模式设置教程，下载文件后双击运行游戏并修改注册表。" * 2,
        "本文介绍南京邮电大学通达学院热门专业和大学排行榜，考生可查看录取分数线。" * 2,
        "悬赏1元已结束，字母圈dom心理和调教方案之后接着介绍DOM节点操作。" * 2,
        "蝶恋花·阅尽天涯离别苦，王国维诗词译文表达相思之情，文章讨论古典文学。" * 2,
    )
    for text in rejected:
        decision = classify_document(text)
        assert not decision["pass"], decision


def test_chunk_v3_rejects_reviewed_false_positive_bodies_and_keeps_real_software() -> None:
    # These are complete bodies (or faithful excerpts preserving the trigger
    # and surrounding domain) from the first diagnostic chunk.  Each passed
    # v2.2 only because a bare term such as 编译/开发工具/命令行 supplied a
    # false software context, or because a strong term overrode page content.
    reviewed_false_positives = {
        "7194_aggregate_blog": (
            "博文<br>【题目】我们公司的数据量非常大,需要的不仅仅是提取数据,要了解SQL方案优化的。"
            "一般在写SQL时需要注意哪些问题,可以提高查询的效率?【解题思路】数据量大的情况下,"
            "不同的SQL语句,消耗的时间相差很大...博文<br>Python 是一种代表简单思想的语言,"
            "其语法相对简单,很容易上手。不过,如果就此小视 Python 语法的精妙和深邃,那就大错特错了。"
            "本文精心筛选了最能展现 Python 语法之精妙的十个知识点,并...博文<br>本文记录自己在自学C++过程中"
            "不同于C的一些知识点,适合于有C语言基础的同学阅读。"
        ),
        "15527_qq_file_listing": (
            "目录 QQUpdate.db 临时文件 rm.swf 动画,可单独删除 SBuddyApplication.dll 互动空间,"
            "这几个同时存在或者删除 SBuddyBase.dll 互动空间,这几个同时存在或者删除 SBuddyCall.exe 互动空间,"
            "这几个同时存在或者删除 SBuddyFriend.dll 互动空间,这几个同时存在或者删除 SendMMS.htm 用 QQ 彩信发送该图片,"
            "删除了也会写入注册表。"
        ),
        "15720_cad_command_instructions": (
            "通过偏移,我们可以做出两个或者多个相互平行的图元。先绘制出一条长度100的直线。"
            "在命令栏中输入偏移命令的快捷方式O,按一下空格键,这时在命令行中提示我们指定偏移距离或者其他操作。"
            "我们输入20,按一下空格键,接着选择要偏移的对象,选择刚才画好的直线,然后选择要偏移到哪一侧。"
            "偏移出来对象后继续选择,如果偏移完成则按一下空格键退出命令。"
        ),
        "17764_incidental_dev_tools": (
            "什么是知识体系?就是把一些零碎的、分散的相对独立的知识概念或观点加以整合,使之形成具有一定联系的知识系统。"
            "学习时需要反思和规划,使用工具如Xmind建立知识体系思维导图,并根据知识的发展不断完善。"
            "很多同学每天刷微信、微博,长期下去会出现碎片化阅读。CSDN有知识图谱,开发者可以建立自己的知识体系。"
            "计划以后建立开发技能图谱、开发书籍图谱、开发工具图谱,以满足开发者入门、进阶的需要。"
        ),
        "18097_psychology_reference_compile": (
            "随着压力源数量上升,儿童出现心理失调的可能性增长。本文讨论心理韧性的创伤修复策略,"
            "从个体、家庭和社区层面说明社会联系、重新定位目标、照顾自己和康复支持。"
            "参考文献包括心理学研究,以及欧文伦道夫帕森,张红超编译(2008):灾难受害者从极度困扰到康复整合之路。"
            "文章全文是心理健康和家庭社会支持的综述,没有技术内容。"
        ),
        "19345_wps_ui_scrollbar": (
            "WPS表格如何制作滚动条?打开WPS表格之后,在菜单里找到开发工具选项,点击开发工具选项,"
            "找到滚动条选项,在表格内画出滚动条,双击进入属性操作界面设置颜色,再在开发工具选项内退出设计。"
            "点击退出设计选项后即可使用这个滚动条。"
        ),
        "18336_browser_recommendation": (
            "下载url是:http://www.google.cn/chrome/browser/desktop/index.html,很多人喜欢他的界面简单,效率高。"
            "火狐浏览器是一款自主内核的浏览器。优点是首页有网站导航、可定制性高、开源软件插件多、兼容性和安全性能较好。"
            "缺点是网页错位、插件过多导致启动速度慢,本文只是浏览器推荐和使用体验。"
        ),
    }
    for label, text in reviewed_false_positives.items():
        decision = classify_document(text)
        assert not decision["pass"], (label, decision)

    software_counterexamples = (
        "Python项目的编译环境需要配置依赖包,开发工具可以调试代码并查看运行日志。" * 3,
        "Linux服务器的系统配置包括服务部署、权限和进程排障,命令行脚本用于检查服务状态。" * 3,
        "MySQL数据库设计需要建立表结构、主键和索引,使用SQL查询语句验证事务和字段。" * 3,
        "Git版本控制管理源代码项目,工作目录中的修改经过签入和签出后由团队审查。" * 3,
        "STM32系统配置需要调用函数写入RTC寄存器,编译器根据源文件构建程序,再通过日志排查时钟同步和中断。" * 3,
    )
    for text in software_counterexamples:
        decision = classify_document(text)
        assert decision["pass"], decision


def test_document_v4_exempts_software_semantics_from_broad_body_vetoes() -> None:
    # ``内存``/``磁盘`` can be ordinary programming or database vocabulary;
    # two independent semantic classes plus a local stack/database context
    # are required before that hardware veto is relaxed.
    c_memory = (
        "C语言函数使用指针和变量访问内存，编译程序后检查数组和字符串的运行结果，"
        "本文解释代码、数据类型与调试方法。"
    ) * 3
    redis_cache = (
        "Redis数据库把数据缓存在内存中，也可以写入磁盘。本文介绍查询、配置和部署缓存，"
        "并说明服务日志、主键和索引在系统中的作用。"
    ) * 3
    assert classify_document(c_memory)["pass"]
    assert classify_document(redis_cache)["pass"]

    # Page words such as 面试/课程 do not override a predominantly technical
    # software article when its local content is independently technical.
    software_interview = (
        "Java面试问题集中讨论类、对象、接口和线程，示例代码说明函数调用、"
        "异常调试与数据库查询，读者可以据此排查运行错误。"
    ) * 3
    linux_course = (
        "Linux操作系统课程设计实现基于socket的服务器程序，项目使用进程、线程和命令行，"
        "通过日志配置网络服务并调试源代码。"
    ) * 3
    assert classify_document(software_interview)["pass"]
    assert classify_document(linux_course)["pass"]

    # ``游戏`` is allowed only when the body is actually software knowledge;
    # employment/product pages remain excluded by the marketing/listing guard.
    game_programming = (
        "Python游戏开发使用模块和框架实现程序逻辑，代码通过函数、变量和调试运行，"
        "并配置依赖包与版本。"
    ) * 3
    generic_game = "热门游戏和网游推荐，玩家可以购买道具、观看直播并分享娱乐体验。" * 3
    employment = (
        "Python就业市场招聘开发者，岗位要求项目经验和本科以上学历，职位薪资与职业生涯信息如下。"
    ) * 3
    product_page = (
        "Python软件产品概述，软件大小和更新时间见产品参数，免费版提供下载，服务热线欢迎咨询。"
    ) * 3
    assert classify_document(game_programming)["pass"]
    assert not classify_document(generic_game)["pass"]
    assert not classify_document(employment)["pass"]
    assert not classify_document(product_page)["pass"]

    # A single illustrative nonsoftware noun must not discard an otherwise
    # technical API article, while a medical page with one incidental HTML
    # mention and a book table of contents remains excluded.
    api_with_food_example = (
        "软件开发中调用API接口和第三方SDK。API可由前端调用后端数据，"
        "SDK提供示例代码、接口规范和开发文档；例如地图SDK可展示美食商家。"
    ) * 3
    incidental_html_medical = (
        "诊所医生介绍患者治疗和手术过程，文章中偶然出现HTML页面标记和软件安装说明。"
        "请咨询医院了解具体情况。"
    ) * 2
    book_catalog = (
        "图书目录 第1章 Linux内核和进程 第2章 文件I/O 第3章 系统调用和库函数，"
        "上册下册内容如下。"
    ) * 3
    windows_firewall = (
        "Windows防火墙配置需要开放端口和设置入站规则，使用命令检查网络连接，"
        "并查看服务器日志排障。"
    ) * 3
    python_install = (
        "Python安装包下载后配置环境，选择版本并运行安装程序，"
        "在Linux和Windows上检查解释器。"
    ) * 3
    assert classify_document(api_with_food_example)["pass"]
    assert not classify_document(incidental_html_medical)["pass"]
    assert not classify_document(book_catalog)["pass"]
    assert classify_document(windows_firewall)["pass"]
    assert classify_document(python_install)["pass"]


def test_document_filter_rejects_code_with_one_comment_and_non_chinese_majority() -> None:
    long_code = "# 这是唯一一条中文注释，下面是代码\n" + (
        "def compute(value):\n    return value + 1\n"
    ) * 100
    english_heavy = (
        "Python software development and configuration documentation with API SDK IDE "
        "build deployment logs and source code. "
    ) * 8 + "这是一段中文说明。"
    assert classify_document(long_code)["reason"] == "not_meaningful_chinese"
    assert classify_document(english_heavy)["reason"] == "not_meaningful_chinese"


def test_stream_collection_workers_preserve_serial_results_and_receipts(tmp_path: Path) -> None:
    collection = tmp_path / "collection.tsv"
    accepted = "Python 软件开发需要清晰的配置、依赖、调试和版本管理说明。" * 3
    rejected = "普通生活页面介绍日常服务和用户体验，不包含软件开发知识。" * 3
    collection.write_bytes(
        (
            "d1\t" + accepted + "\n"
            "d1\t" + accepted + "\n"
            "d2\t" + rejected + "\n"
            "d2\t" + rejected + "\n"
            "\n"
            "malformed-without-tab\n"
            "d3\t" + accepted + "\n"
        ).encode("utf-8")
    )
    serial = _stream_collection(collection, {"d1", "d2"}, workers=1)
    parallel = _stream_collection(collection, {"d1", "d2"}, workers=2, batch_size=2)
    assert parallel == serial
    assert parallel[1] == {"d1": "duplicate_document_id", "d2": "not_meaningful_chinese"}
    assert parallel[2]["duplicate_ids"] == 1
    assert parallel[2]["retained"] == 2


def test_grouping_connects_shared_positive_and_numeric_templates_but_not_ids_alone() -> None:
    eligible = {}
    for key, source_id, text, doc_id in (
        ("dev:q1", "q1", "Python 3 安装配置", "d1"),
        ("dev:q2", "q2", "Python 4 安装配置", "d2"),
        ("dev:q3", "q3", "Linux 服务器配置", "d3"),
    ):
        eligible[key] = {"query": _query(key, source_id, text), "positive_ids": [doc_id], "judgments": [], "grades": {doc_id: 2}}
    documents = {f"d{i}": _doc(f"Python 软件开发需要清晰的配置、依赖、调试和版本管理说明，文档编号{i}。") for i in range(1, 4)}
    groups, metadata = group_queries(eligible, documents)
    q1_q2 = [members for members in groups.values() if {"dev:q1", "dev:q2"} <= set(members)]
    assert len(q1_q2) == 1
    assert all(not ({"dev:q1", "dev:q2", "dev:q3"} <= set(members)) for members in groups.values())
    assert any("query_template" in metadata[group]["reasons"] for group, members in groups.items() if {"dev:q1", "dev:q2"} <= set(members))


def test_small_offline_release_preserves_binary_qrels_source_grades_and_split(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    data = raw / "data"
    data.mkdir(parents=True)
    labels = []
    for index in range(60):
        value, suffix = index, []
        while True:
            suffix.append(chr(65 + value % 26))
            value = value // 26 - 1
            if value < 0:
                break
        labels.append("".join(reversed(suffix)))
    (data / "queries.dev.tsv").write_text(
        "".join(f"q{index}\tPython 安装配置 topic{labels[index - 1]}\n" for index in range(1, 61)),
        encoding="utf-8",
    )
    qrel_rows = [
        "q1\t0\td1\t2",
        "q1\t0\td61\t3",
        "q1\t0\td2\t1",
        "q1\t0\tmissing\t0",
        "q2\t0\td2\t3",
    ] + [f"q{index}\t0\td{index}\t2" for index in range(3, 61)]
    (data / "qrels.dev.tsv").write_text("\n".join(qrel_rows) + "\n", encoding="utf-8")
    body_one = "Python 软件开发需要清晰的配置、依赖、调试和版本管理说明。本文介绍开发环境、代码运行、错误排查和测试流程。" * 2
    body_two = "Linux 服务器配置需要管理网络服务、用户权限、日志和系统安全策略。本文说明服务部署、命令行排障以及进程监控方法。" * 2
    collection_rows = [
        "d1\t" + body_one,
        "d2\t" + body_two,
    ] + [
        f"d{index}\tPython 软件开发需要清晰的配置、依赖、调试和版本管理说明。本文介绍开发环境、代码运行和错误排查，文档编号{index}。" * 2
        for index in range(3, 61)
    ]
    collection_rows.append("d61\t" + body_one)
    (data / "collection.tsv").write_text("\n".join(collection_rows) + "\n", encoding="utf-8")
    collection_bytes = (data / "collection.tsv").read_bytes()
    monkeypatch.setattr(preparer, "COLLECTION_BYTES", len(collection_bytes))
    monkeypatch.setattr(preparer, "COLLECTION_LFS_SHA256", hashlib.sha256(collection_bytes).hexdigest())
    release = tmp_path / "release"
    evidence = tmp_path / "evidence"
    manifest = prepare_release(
        raw,
        release,
        evidence,
        splits=("dev",),
        minimum_queries=60,
        minimum_documents=2,
    )
    assert manifest["num_queries"] == 60
    qrels = json.loads((release / "qrels.json").read_text(encoding="utf-8"))
    assert qrels["q1"] == {"d1": 1}
    assert qrels["q2"] == {"d2": 1}
    assert qrels["q3"] == {"d3": 1}
    judgments = [json.loads(line) for line in (release / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["original_grade"] for row in judgments} == {1, 2, 3}
    assert any(row["source_doc_id"] == "d61" and row["canonical_doc_id"] == "d1" and row["original_grade"] == 3 for row in judgments)
    queries = [json.loads(line) for line in (release / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["metadata"]["original_split"] == row["metadata"]["source_split"] == "dev" for row in queries)
    assert sum(row["metadata"]["sampling_preview_split"] == "dev" for row in queries) == 10
    assert sum(row["metadata"]["sampling_preview_split"] == "test" for row in queries) == 50
    assert sum(row["metadata"]["sampling_preview_smoke"] for row in queries) == 1
    selection = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
    assert selection["counts"]["preview_queries"] == 60
    provenance = json.loads((evidence / "source_provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_document_count"] == 61
    assert provenance["canonical_document_count"] == 60
    assert provenance["exact_alias_document_count"] == 1
    original = [json.loads(line) for line in (evidence / "original_judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["source_doc_id"] == "missing" and row["canonical_doc_id"] is None for row in original)
    assert any(row["source_doc_id"] == "d61" and row["canonical_doc_id"] == "d1" and row["original_grade"] == 3 for row in original)
    assert all(row["doc_id"] != "missing" for row in judgments)
    loaded = load_bundle(release)
    assert loaded["documents"] == [json.loads(line) for line in (release / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert loaded["documents"] and all(row["title"] == "" for row in loaded["documents"])
    assert loaded["queries"] == queries
    assert loaded["qrels"] == qrels
    assert loaded["judgments"] == judgments
    assert verify_release(release)["state"] == "labels_ready"
    status = json.loads((evidence / "status.json").read_text(encoding="utf-8"))
    assert status["published_state"] == "labels_ready"
    assert status["round_trip_audit"]["audited_fields_equal"] is True
    suite = tmp_path / "export"
    export_receipt = export_dci(release, suite, "software_zh", io_workers=1)
    assert export_receipt["task"] == "software_zh"
    assert export_receipt["num_queries"] == 60
    exported_queries = [json.loads(line) for line in (suite / "metadata" / "software_zh" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exported_queries and all("group_id" in row for row in exported_queries)
    exported_corpus = [json.loads(line) for line in (suite / "metadata" / "software_zh" / "corpus_files.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exported_corpus and all(row["bytes"] > 0 for row in exported_corpus)


def test_expected_small_files_merges_dev_train_and_complete_acquisition_receipts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "small-files-receipt.json").write_text(json.dumps({
        "files": [{"path": "data/queries.dev.tsv", "bytes": 1, "sha256": "old-dev"}],
    }), encoding="utf-8")
    (raw / "train-acquisition-receipt.json").write_text(json.dumps({
        "files": [{"path": "data/queries.train.tsv", "bytes": 2, "sha256": "train"}],
    }), encoding="utf-8")
    (raw / "acquisition-receipt.json").write_text(json.dumps({
        "files": [
            {"path": "data/queries.dev.tsv", "bytes": 3, "sha256": "complete-dev"},
            {"path": "data/queries.train.tsv", "bytes": 4, "sha256": "complete-train"},
        ],
    }), encoding="utf-8")
    expected = _expected_small_files(raw)
    assert expected["data/queries.dev.tsv"]["sha256"] == "complete-dev"
    assert expected["data/queries.train.tsv"]["sha256"] == "complete-train"


def test_source_card_url_is_pinned_to_revision() -> None:
    assert preparer.SOURCE_URLS["dataset_card"].endswith(
        f"/blob/{preparer.REVISION}/README.md"
    )
    assert "/blob/main/" not in preparer.SOURCE_URLS["dataset_card"]


def test_incomplete_collection_part_is_never_read_or_renamed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    data = raw / "data"
    data.mkdir(parents=True)
    partial = data / "collection.tsv.part"
    partial.write_bytes(b"do not touch")
    try:
        prepare_release(raw, tmp_path / "release", tmp_path / "evidence", splits=("dev",))
    except PreparationBlocked as exc:
        assert "collection.tsv" in str(exc)
    else:
        raise AssertionError("an incomplete collection must block")
    assert partial.read_bytes() == b"do not touch"
