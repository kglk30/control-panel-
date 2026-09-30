import os
import sys
import ctypes

def main():
    qnn_dir = r'C:\Users\DELL IN14 5441 X1P\Downloads\projects\control-panel\.venv\Lib\site-packages\onnxruntime_qnn'
    os.environ['PATH'] = qnn_dir + ';' + os.environ['PATH']

    print("1. Loading QnnSystem.dll...")
    try:
        qnn_sys = ctypes.CDLL(os.path.join(qnn_dir, 'QnnSystem.dll'))
        print("   SUCCESS.")
    except Exception as e:
        print("   FAILED:", e)
        sys.exit(1)

    print("2. Loading QnnHtp.dll...")
    try:
        qnn_htp = ctypes.CDLL(os.path.join(qnn_dir, 'QnnHtp.dll'))
        print("   SUCCESS.")
    except Exception as e:
        print("   FAILED:", e)
        sys.exit(1)

    print("3. Resolving QnnSystemInterface_getProviders...")
    get_sys_providers = getattr(qnn_sys, 'QnnSystemInterface_getProviders', None)
    if not get_sys_providers:
        print("   FAILED: Export not found.")
        sys.exit(1)
    
    get_sys_providers.restype = ctypes.c_int
    get_sys_providers.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_uint32)]
    print("   SUCCESS.")

    print("4. Resolving QnnInterface_getProviders...")
    get_htp_providers = getattr(qnn_htp, 'QnnInterface_getProviders', None)
    if not get_htp_providers:
        print("   FAILED: Export not found.")
        sys.exit(1)
        
    get_htp_providers.restype = ctypes.c_int
    get_htp_providers.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_uint32)]
    print("   SUCCESS.")

    print("5. Calling QnnSystemInterface_getProviders...")
    sys_providers_ptr = ctypes.POINTER(ctypes.c_void_p)()
    sys_num_providers = ctypes.c_uint32(0)
    res_sys = get_sys_providers(ctypes.byref(sys_providers_ptr), ctypes.byref(sys_num_providers))
    print(f"   SUCCESS. Result: {res_sys}, Count: {sys_num_providers.value}")

    print("6. Calling QnnInterface_getProviders...")
    htp_providers_ptr = ctypes.POINTER(ctypes.c_void_p)()
    htp_num_providers = ctypes.c_uint32(0)
    res_htp = get_htp_providers(ctypes.byref(htp_providers_ptr), ctypes.byref(htp_num_providers))
    print(f"   SUCCESS. Result: {res_htp}, Count: {htp_num_providers.value}")
    
    if htp_num_providers.value > 0 and htp_providers_ptr:
        print("7. Reading QNN API Version from Interface Struct...")
        # providers_ptr is a pointer to an array of pointers to QnnInterface_t
        first_provider = htp_providers_ptr[0]
        # first_provider points to QnnInterface_t. First 8 bytes are Qnn_ApiVersion_t (two uint32_t)
        core_api_version = ctypes.cast(first_provider, ctypes.POINTER(ctypes.c_uint32))[0]
        backend_api_version = ctypes.cast(first_provider, ctypes.POINTER(ctypes.c_uint32))[1]
        
        core_major = core_api_version >> 16
        core_minor = core_api_version & 0xFFFF
        backend_major = backend_api_version >> 16
        backend_minor = backend_api_version & 0xFFFF
        
    import psutil
    p = psutil.Process(os.getpid())
    modules = [m.path.lower() for m in p.memory_maps()]
    soc_loaded = any("qcsocserviceutils.dll" in m for m in modules)
    print(f"8. Is QcSoCServiceUtils.dll loaded? {soc_loaded}")
    
    print("Last successful native call: Reading API Version from Interface Struct.")
    print("Process exiting normally.")
    sys.exit(0)

if __name__ == '__main__':
    main()
