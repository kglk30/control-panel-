import onnx

MODEL_PATH = r"C:\Users\DELL IN14 5441 X1P\Downloads\projects\control-panel\models\mobilenetv2\mobilenetv2_qnn_htp_extracted\job_jp2r4vnqg_optimized_onnx\model.onnx"

m = onnx.load(MODEL_PATH, load_external_data=False)

print("IR version:", m.ir_version)
print("Producer:", m.producer_name, m.producer_version)
print("Opset imports:", [(o.domain, o.version) for o in m.opset_import])
print()

for node in m.graph.node:
    print("Node name:", node.name)
    print("Node op_type:", node.op_type)
    print("Node domain:", node.domain)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.STRING:
            print(f"  attr {attr.name} = {attr.s}")
        elif attr.type == onnx.AttributeProto.INT:
            print(f"  attr {attr.name} = {attr.i}")
        else:
            print(f"  attr {attr.name} (type {attr.type})")
    print()
