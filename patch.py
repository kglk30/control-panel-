
import struct
exe = r'C:\Users\DELL IN14 5441 X1P\Downloads\projects\control-panel\.venv\Scripts\python.exe'
with open(exe, 'r+b') as f:
    dos = f.read(64)
    pe_offset = struct.unpack('<I', dos[60:64])[0]
    f.seek(pe_offset + 24 + 72)
    f.write(struct.pack('<Q', 16 * 1024 * 1024))
print('Patched shim')

