"""Targeted contracts for the frozen-v5 T2Ranking refinement."""

from __future__ import annotations

import hashlib

from scripts import refine_t2ranking_zh_20261102 as refine


def test_structural_nonsoftware_pages_are_rejected_before_incidental_stack_terms() -> None:
    rejected = (
        "Minecraft Java版和基岩版的方块、红石元件、命名空间ID及生物介绍。" * 4,
        "大学年鉴目录一、学校概况 二、院系设置 三、专业设置 四、课程表。" * 4,
        "软件价格优惠购买，扫码联系客服，推荐电脑产品并提供优惠券和报价。" * 4,
        "刑事诉讼案件的法律法规和法院审理流程，本文讨论证据与辩护。" * 4,
        "Python开发自测题（含答案） 单选题 A、函数 B、变量 C、类 D、对象。" * 4,
        "Windows普通用户重装系统、播放视频和安装驱动的操作步骤。" * 4,
    )
    for body in rejected:
        result = refine.classify_document(body)
        assert not result["pass"], (result["reason"], body[:40])


def test_software_semantics_survive_without_query_or_metadata_whitelisting() -> None:
    accepted = (
        "MySQL数据库创建数据表，设置主键和索引，使用SELECT查询语句检查表结构并执行事务。" * 3,
        "Linux系统服务排障需要查看日志和进程状态，通过命令行修改权限并部署服务。" * 3,
        "Python的闭包和匿名函数会捕获变量，函数参数与返回值可以通过代码测试和调试验证。" * 3,
        "C语言的数据类型、指针、数组和结构体需要在编译器中声明，程序通过函数调用读写变量。" * 3,
        "hosts文件把域名解析到127.0.0.1，编辑并保存配置后可以屏蔽指定域名。" * 3,
    )
    for body in accepted:
        result = refine.classify_document(body)
        assert result["pass"], result
        assert result["chinese_ratio"] >= 0.35
        assert result["content_chars"] >= 80


def test_positive_document_rejection_excludes_the_whole_query_without_shortening_gold() -> None:
    good = {
        "_id": "good",
        "language": "zh",
        "metadata": {"source": "source", "original_id": "good"},
        "text": "MySQL数据库创建数据表，设置主键和索引，使用SELECT查询语句检查表结构并执行事务。" * 3,
        "title": "",
    }
    bad = {
        "_id": "bad",
        "language": "zh",
        "metadata": {"source": "source", "original_id": "bad"},
        "text": "Minecraft Java版和基岩版的方块、红石元件、命名空间ID及生物介绍。" * 4,
        "title": "",
    }
    q1 = {"_id": "q1", "language": "zh", "metadata": {"original_split": "dev"}, "text": "数据库索引"}
    q2 = {"_id": "q2", "language": "zh", "metadata": {"original_split": "train"}, "text": "游戏方块"}
    judgment_good = {"query_id": "q1", "doc_id": "good", "original_grade": 2, "source": "human"}
    judgment_bad = {"query_id": "q2", "doc_id": "bad", "original_grade": 2, "source": "human"}
    bundle = {
        "documents": [good, bad],
        "queries": [q1, q2],
        "qrels": {"q1": {"good": 1}, "q2": {"good": 1, "bad": 1}},
        "judgments": [judgment_good, judgment_bad],
        "evidence": [],
    }
    documents, queries, qrels, judgments, _, aux = refine._select_rows(bundle)
    assert documents == [good]
    assert queries == [q1]
    assert qrels == {"q1": {"good": 1}}
    assert judgments == [judgment_good]
    assert any(row["query_id"] == "q2" and row["decision"] == "exclude" for row in aux["query_decisions"])


def test_hash_sample_is_deterministic_and_contains_fifty_rows() -> None:
    rows = [
        {"_id": str(index), "text": "Linux系统配置和服务排障需要查看日志与进程。" * 3}
        for index in range(60)
    ]
    bundle = {"documents": rows}
    decisions = {row["_id"]: refine.classify_document(row["text"]) for row in rows}
    sample = refine._audit_sample(bundle, decisions)
    expected = sorted((row["_id"] for row in rows), key=lambda value: hashlib.sha256(value.encode()).hexdigest())[:50]
    assert len(sample) == 50
    assert [row["doc_id"] for row in sample] == expected
    assert all(row["preview"] for row in sample)


def test_v7_rejects_the_reviewed_false_positive_surfaces() -> None:
    cases = {
        "classical_grammar": "最佳回答 文言文中的主谓倒装就是谓语放在主语前面，英语倒装句也有类似语法。帮我找出例句。" * 3,
        "career_advice": "工作时的心情很重要，熟悉Linux和代码可以找不到工作吗？单片机开发编程应用、职位和薪资建议。" * 3,
        "mixed_note_ad": "初级问题：Flow Chart是什么意思？来自度娘。Python从入门到实践全套笔记，今天分享237页超详细笔记，另有C++课程目录。" * 3,
        "navigation_listing": "首页Web开发编程语言数据库移动开发计算机基础运维标签登录 标签列表 单片机数码管应用 代码详解。" * 3,
        "code_fragment_listing": "C语言猜数字小游戏，方法一：不用循环比较三次。#include int main(){代码片段} 方法二：另一个循环队列片段。" * 3,
        "answer_board": "问题点数:40分，结帖人，题目是自动判断输入数字个数并调用函数，请大家指教。import java.util.Scanner; public class Test {}" * 3,
        "duplicated_registry": "映射的网络驱动器在一段时间自动断开，修改注册表编辑键值增加超时期限，方法一如下。" * 3,
    }
    for name, body in cases.items():
        decision = refine.classify_document(body)
        assert not decision["pass"], (name, decision)


def test_v7_keeps_concrete_software_and_system_topics_with_hardware_words() -> None:
    accepted = (
        "PHP字符串定义和函数参数，程序运行时内存大小只影响字符串长度，使用语法和变量进行编码。" * 3,
        "C语言数据类型包括指针、数组和结构体，变量在编译时分配存储单元，函数调用返回结果。" * 3,
        "Windows服务宿主svchost.exe是系统进程，在任务管理器查看进程和服务状态并排障。" * 3,
        "HTML网页中使用img标签插入图片，标签和属性决定文档结构，浏览器加载网页文件。" * 3,
        "AWS Lambda函数用Python创建执行角色和API配置，部署后测试函数并查看日志。" * 3,
    )
    for body in accepted:
        decision = refine.classify_document(body)
        assert decision["pass"], decision
