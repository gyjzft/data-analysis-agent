"""
Gradio Web 界面
提供用户友好的交互界面
"""
import os
import json
import gradio as gr
import pandas as pd
from src.config import Config
from src.agent.core import DataAnalysisAgent
from src.tools import modeling


# 全局 Agent 实例
agent = DataAnalysisAgent()

# 用于校验聊天消息格式的组件实例（Gradio 6.x 校验用）
_chat_validator = gr.Chatbot()


def _sanitize_history(history):
    """确保聊天记录是 Gradio 能接受的格式，格式非法时降级为纯文本。
    这样即使某个消息有问题，也只是在聊天里显示一行文字，而不是触发 Gradio 的错误弹窗盖住聊天。
    """
    sanitized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        # 用 Gradio 自己的校验器验证这条消息
        try:
            _chat_validator._postprocess(msg)
            sanitized.append(msg)
            continue
        except Exception:
            pass
        # 校验失败 → 降级为纯文本
        role = msg.get("role", "assistant")
        content = msg.get("content")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    texts.append(block)
            fallback = "\n\n".join(t for t in texts if t) or "（消息格式异常，已降级为文本）"
        else:
            fallback = str(content) if content else "（消息内容为空）"
        sanitized.append({"role": role, "content": fallback})
    return sanitized


def handle_file_upload(file):
    """处理文件上传"""
    if file is None:
        return "请先上传数据文件", gr.update(visible=False), gr.update(visible=False)

    result = agent.load_data(file.name)

    if agent.df is not None:
        preview = agent.df.head(10).to_markdown(index=False)
        return result + "\n\n数据预览（前10行）:\n" + preview, gr.update(visible=True), gr.update(visible=True)
    return result, gr.update(visible=False), gr.update(visible=False)


def handle_chat(message, history):
    """处理用户消息"""
    # 兼容 Gradio 6.x 的 Chatbot 格式：使用 role/content 字典
    history = history or []
    new_history = list(history)

    # 确保 history 里的旧格式也转换成字典格式
    normalized = []
    for item in new_history:
        if isinstance(item, dict):
            # content 可能是字符串（普通文本）或列表（内容块），原样保留
            normalized.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized.append({"role": "user", "content": str(item[0])})
            normalized.append({"role": "assistant", "content": str(item[1])})
    new_history = normalized

    if agent.df is None:
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": "请先上传数据文件"})
        return _sanitize_history(new_history)

    try:
        response = agent.chat(message)
        text = response["text"]

        # 用内容块格式：文字 + 图片（如果有）
        content = [{"type": "text", "text": text}]
        if response.get("image"):
            img_path = response["image"]
            # 用绝对路径，避免相对路径解析问题
            if os.path.exists(img_path):
                # Gradio 6.x 的 Chatbot 用 path 字段渲染图片（会自动识别 mime 类型）
                content.append({"path": img_path})

        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": content})
        return _sanitize_history(new_history)
    except Exception as e:
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": f"错误: {str(e)}"})
        return _sanitize_history(new_history)


# 预测面板状态（全局）
_current_model_id = None
_current_features = []   # [{"name","type","categories"}]

# 预建输入框槽位数（留足余量）
MAX_SLOTS = 20


def load_model_panel(model_id):
    """加载模型，读取特征信息，动态显示对应的输入框"""
    global _current_model_id, _current_features
    model_id = (model_id or "").strip()
    if not model_id:
        return "请输入模型ID", None

    try:
        meta = modeling.load_model_artifacts(model_id)["meta"]
    except Exception as e:
        return f"加载失败: {str(e)}", None

    _current_model_id = model_id
    _current_features = meta["features"]

    # 生成 40 个更新（每个槽位：Number 和 Dropdown 各一个）
    updates = []
    for i in range(MAX_SLOTS):
        if i < len(_current_features):
            f = _current_features[i]
            if f["type"] == "numeric":
                updates += [
                    gr.update(visible=True, label=f["name"], value=None),
                    gr.update(visible=False, label=""),
                ]
            else:
                updates += [
                    gr.update(visible=False, label=""),
                    gr.update(visible=True, label=f["name"], choices=f["categories"], value=None),
                ]
        else:
            updates += [
                gr.update(visible=False, label=""),
                gr.update(visible=False, label=""),
            ]

    info = (
        f"✅ 模型已加载（{meta['task_type']}，目标列: {meta['target']}，"
        f"最佳模型: {meta['best_model']}）\n"
        f"请填写以下 {len(_current_features)} 个特征后点击预测："
    )
    return info, *updates


def do_predict(*input_values):
    """收集输入框的值，调用保存的模型预测"""
    if not _current_model_id or not _current_features:
        return "请先加载模型"

    values = {}
    missing = []
    for i, f in enumerate(_current_features):
        num_val = input_values[2 * i]
        cat_val = input_values[2 * i + 1]
        if f["type"] == "numeric":
            v = num_val
        else:
            v = cat_val
        if v is None or v == "":
            missing.append(f["name"])
        values[f["name"]] = v

    if missing:
        return f"请填写以下特征: {', '.join(missing)}"

    try:
        pred, meta = modeling.predict_from_values(_current_model_id, values)
    except Exception as e:
        return f"预测失败: {str(e)}"

    if meta["task_type"] == "classification":
        return f"🎯 预测结果（{meta['target']}）: **{pred}**"
    else:
        return f"📈 预测结果（{meta['target']}）: **{pred:,.2f}**"


def close_predict_panel():
    """关闭预测面板并清空状态"""
    global _current_model_id, _current_features
    _current_model_id = None
    _current_features = []
    updates = []
    for i in range(MAX_SLOTS):
        updates += [
            gr.update(visible=False, label=""),
            gr.update(visible=False, label=""),
        ]
    return "预测面板已关闭", *updates


def clear_all():
    """清空所有状态"""
    global agent
    agent = DataAnalysisAgent()
    updates = []
    for i in range(MAX_SLOTS):
        updates += [
            gr.update(visible=False, label=f"特征 {i+1}"),
            gr.update(visible=False, label=f"特征 {i+1}"),
        ]
    return (
        "", [], None, "已清空，请重新上传数据",
        gr.update(visible=False), gr.update(visible=False),
        "请先输入模型ID并点击「加载模型」", *updates,
    )


# 构建 Gradio 界面
with gr.Blocks(title="数据分析 AutoAgent") as demo:
    gr.Markdown("""
    # 📊 数据分析 AutoAgent
    上传 CSV/Excel 数据文件，用自然语言描述你的分析需求
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 左侧：上传 + 数据预览
            file_input = gr.File(label="📁 上传数据文件 (CSV/Excel)", type="filepath")
            upload_btn = gr.Button("上传并加载", variant="primary")
            data_status = gr.Markdown("等待上传数据...")
            chat_box = gr.Chatbot(label="💬 对话", height=400, visible=False)
            clear_btn = gr.Button("🗑️ 清空重新开始")

        with gr.Column(scale=1):
            # 右侧：分析对话
            msg_box = gr.Textbox(label="📝 你的分析指令",
                                 placeholder="例如: 帮我分析数据的分布情况",
                                 lines=3)
            send_btn = gr.Button("发送", variant="primary")

            gr.Markdown("""
            ### 💡 可以尝试的指令
            - 帮我看看数据的描述统计
            - 画一个收入的直方图
            - 分析各列之间的相关性
            - 检验收入是否服从正态分布
            - 比较不同地区的收入差异
            - 建立一个回归模型预测消费
            - 对数据进行聚类分析
            """)

    # 预测区域：加载模型 → 自动生成特征输入框 → 预测
    with gr.Row(visible=False) as predict_section:
        with gr.Column():
            gr.Markdown("""
            ### 🔮 模型预测
            输入模型ID → 加载模型 → 自动生成特征输入框 → 填写后预测
            """)
            with gr.Row():
                model_id_input = gr.Textbox(label="模型ID",
                                            placeholder="例如: regression_salary_6f4b7a",
                                            scale=3)
                load_btn = gr.Button("加载模型", variant="primary", scale=1)

            # 20 个输入框槽位（每个槽位 = 数字框 + 下拉框，按特征类型显示其一）
            num_inputs = []
            cat_inputs = []
            for row_i in range(0, MAX_SLOTS, 2):
                with gr.Row():
                    for j in range(2):
                        i = row_i + j
                        with gr.Column(scale=1):
                            num = gr.Number(visible=False, label=f"特征 {i+1}")
                            cat = gr.Dropdown(choices=[], visible=False, label=f"特征 {i+1}")
                            num_inputs.append(num)
                            cat_inputs.append(cat)

            # 交错顺序：每个槽位 [数字框, 下拉框]，与 do_predict 的读取方式一致
            interleaved_inputs = []
            for i in range(MAX_SLOTS):
                interleaved_inputs += [num_inputs[i], cat_inputs[i]]

            with gr.Row():
                predict_btn = gr.Button("预测", variant="primary")
                close_btn = gr.Button("关闭预测")
            predict_output = gr.Markdown("请先输入模型ID并点击「加载模型」")

    # 绑定事件
    upload_btn.click(
        fn=handle_file_upload,
        inputs=[file_input],
        outputs=[data_status, chat_box, predict_section],
    )

    send_btn.click(
        fn=handle_chat,
        inputs=[msg_box, chat_box],
        outputs=[chat_box],
    ).then(lambda: "", outputs=[msg_box])

    msg_box.submit(
        fn=handle_chat,
        inputs=[msg_box, chat_box],
        outputs=[chat_box],
    ).then(lambda: "", outputs=[msg_box])

    load_btn.click(
        fn=load_model_panel,
        inputs=[model_id_input],
        outputs=[predict_output] + interleaved_inputs,
    )

    predict_btn.click(
        fn=do_predict,
        inputs=interleaved_inputs,
        outputs=[predict_output],
    )

    close_btn.click(
        fn=close_predict_panel,
        outputs=[predict_output] + interleaved_inputs,
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            msg_box, chat_box, file_input, data_status, chat_box,
            predict_section, predict_output,
        ] + interleaved_inputs,
    )


def launch():
    """启动服务"""
    if not Config.validate():
        return
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch()
