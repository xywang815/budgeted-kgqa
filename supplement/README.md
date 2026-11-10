# SUPP-20260909 固定补充实验协议

冻结日期 2026-09-09，实验产生任何答案之前。此协议是看到旧实验和第一轮审稿后的后续验证，不是整个研究事前注册。

## 目的与假设

检验在相同初始图、相同候选装填池和相同 256 evidence-token 上限下，完整 RACPS 是否优于简单关系/成本排序，以及冻结的通用语义编码器。重点是识别边界，不预设胜者。

## 数据和划分

使用官方 WebQSP test v1.0 和既有 UniK-QA 资源审计，沿用有效 parse、有子图且 topic 出现的资格标准。总数必须为 1,577。旧推理的 500 个 ID 标记 legacy500，余下 1,077 标记 extension1077。全部题都运行五种条件；主要新推断限定 extension1077，legacy500 和全量仅作为描述性完整报告。旧结果已见，不能把旧 500 题称作新的留出集。官方 1-chain/2+-chain 是异质性分层。

## 固定管线

- 初始路径枚举复用冻结代码，outgoing-only，深度 1 和 2，最多 100,000 条；topic 来自官方解析，gold 答案不用于选择。
- 共同装填候选池最多 1,000 条。取 full-path BM25 前 500 和 relation BM25 前 500 的有序并集，再按 full-path BM25 补足到 1,000。该上限为计算预算选择，不根据测试效果调参。评分语料为完整枚举池，排序、平局规则写入源码。
- 五个条件为 BM25、RACPS（lambda .9，gamma 1）、relation-density（lambda 1，gamma 1）、E5-rank、E5-density。前两种词汇分数使用原代码；E5 输入相同完整路径文本，官方 query/passsage 前缀、masked mean pooling、L2 normalization、512 token 截断；E5-density 为非负 cosine 分数除以独立行成本，E5-rank 不作成本归一化。
- 编码器 intfloat/e5-small-v2，revision ffb93f3bd4047442299a41ebb6fa998a38507c52，冻结 FP32，不作任务训练。
- Qwen/Qwen3-8B-AWQ，revision cb7d6a337aadb4d2082ed0dcef1032e4f8645194，既有 AWQ/FP16 环境。一次 greedy 生成、max_new_tokens=32、repetition_penalty=1.10、no_repeat_ngram_size=3、关闭 thinking。32 与旧 WebQSP 的实际运行命令一致；MetaQA 是 96。
- 证据行中答案代码直接计入每次装填的完整 tokenizer 检查，预算 256。所有条件采用同一实现，不执行旧协议的先装填再截尾。候选单行成本使用 A1 标签，最终接受必须满足完整代码证据不超预算。
- 生成条件顺序按问题编号循环平移，缓解固定执行顺序的时间偏差；不改变文本顺序或预算。保存模型输出、代码映射、路径、ID、候选统计、每项时间和精确 token 数。
- 非法 A-code 作为错误预测保留在分母；无代码视为空回答。端点直接输出作为零额外阅读器调用的参考；oracle 不参与新实验。

## 统计

主要新家族为 extension1077 上 RACPS 相对 BM25、relation-density、E5-rank 的三项宏 F1 双侧配对差异，联合 Holm 校正。固定统计 seed=20260909，20,000 次按官方 chain strata 的成对 bootstrap，50,000 次逐题 sign-flip。95% 区间为未校正区间。E5-density、各 chain、候选量、legacy500 和总体的比较为探索性或描述性，清楚标记；不重定义主家族。无需所有比较显著才报告结果。

## 运行边界和失败处理

最多 7,885 次正式生成。只能为异常中断恢复尚未保存的条件，不替换已保存正常结果。先用独立的人工小图做 token/评分/生成 smoke，不查看测试得分来调参。GPU 使用前检查；不停止其他项目。服务器所有新增文件在项目 revision_20260909 内，旧源与结果不改动。断点文件逐题原子更新。最长 6 小时工程上限，超时如实记录部分完成并恢复未完成任务，不能报告为全量。

不在正文声称此协议已外部注册。最终源代码、依赖、模型版本、协议和输出均作 SHA256 清单。
