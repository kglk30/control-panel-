import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

ep_lib_path = qnn_ep.get_library_path()
ort.register_execution_provider_library("QNNExecutionProvider", ep_lib_path)

all_ep_devices = ort.get_ep_devices()

for i, d in enumerate(all_ep_devices):
    print(f"--- Device {i} ---")
    print("ep_name:", d.ep_name)
    print("ep_vendor:", getattr(d, "ep_vendor", None))
    print("ep_metadata:", getattr(d, "ep_metadata", None))
    print("device (hardware):", getattr(d, "device", None))
    try:
        dev = d.device
        print("  device.type:", getattr(dev, "type", None))
        print("  device.vendor:", getattr(dev, "vendor", None))
        print("  device.metadata:", getattr(dev, "metadata", None))
    except Exception as e:
        print("  (could not read device sub-fields:", e, ")")
    print()
