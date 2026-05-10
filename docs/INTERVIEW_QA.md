# 常见面试问答

## Q1：这个项目和普通日志脚本有什么区别？

A：普通脚本通常只解析或画图。本项目把解析、统计、诊断、建议、绘图、报告和可选问答组织成一个 Agent 工作流，并提供 API、前端、测试和 Docker。

## Q2：为什么说它是 Agent？

A：它根据输入日志和用户问题，自动调用多个工具并聚合结果。Agent 的核心不是必须有复杂 planner，而是能把任务分解成工具执行链。

## Q3：为什么这个项目没有 RAG？

A：Training Log Agent 的核心输入是训练日志，关键挑战是结构化解析和规则诊断，不是知识检索。因此它更适合定位为 Rule-based / Tool Agent。后续可以加入调参知识库 RAG。

## Q4：日志解析如何保证鲁棒性？

A：通过 metric alias、正则容错、百分比归一化、缺失字段跳过和多格式 parser。测试覆盖了 sample log、百分比、缺失字段、CSV 和中文日志格式。

## Q5：诊断规则从哪里来？

A：来自深度学习训练经验，例如过拟合看 train/val loss gap，震荡看 recent metric std，类别不平衡看 leaf/stem IoU gap，PR gap 用于判断漏检或误检倾向。

## Q6：LLM 在项目中做什么？

A：LLM 只用于回答用户附加问题和自然语言解释。核心指标和诊断不依赖 LLM，默认 mock provider 保证 Demo 稳定。

## Q7：如何支持真实 OpenAI/DeepSeek/Qwen？

A：项目中 `OpenAICompatibleProvider` 已封装 chat completion 请求，通过环境变量配置 API Key、base URL 和 model 即可扩展。

## Q8：后续最值得优化什么？

A：多实验对比、实验 registry、TensorBoard/W&B/MLflow 支持、RAG 调参知识库、HTML/PDF 报告导出和规则阈值配置化。
