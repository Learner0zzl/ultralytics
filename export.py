from my_utils import find_files_by_ext
# 输入固定为images   输出固定为output  下面方法已通过推理验证
import onnx
from onnx import helper, shape_inference


root_dir = r"F:\Work\Honesort\Color\C1\C1_640"
onnx_paths = find_files_by_ext(root_dir, ".onnx", "path")
for idx, onnx_path in enumerate(onnx_paths):
    print(f"{idx + 1}/{len(onnx_paths)} {onnx_path}")
    # 加载模型
    model = onnx.load(onnx_path)

    # 保存原始输入输出名称
    original_input_name = model.graph.input[0].name
    original_output_name = model.graph.output[0].name

    # 修改输入名称
    model.graph.input[0].name = "images"

    # 修改输出名称
    model.graph.output[0].name = "output"

    # 更新所有节点中引用原始名称的地方
    for node in model.graph.node:
        # 更新输入中引用原始输出名称的地方
        for i, input_name in enumerate(node.input):
            if input_name == original_output_name:
                node.input[i] = "output"
        # 更新输出中引用原始输出名称的地方
        for i, output_name in enumerate(node.output):
            if output_name == original_output_name:
                node.output[i] = "output"

    # 重新进行形状推断以确保一致性
    inferred_model = shape_inference.infer_shapes(model)

    # 检查模型
    onnx.checker.check_model(inferred_model, full_check=True)

    # 保存修改后的模型
    output_path = onnx_path.replace('.onnx', '_renamed.onnx')
    onnx.save(inferred_model, output_path)
    print(f"已保存修改后的模型: {output_path}")
