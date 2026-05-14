import winreg

paths = []
try:
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
    paths.extend(winreg.QueryValueEx(key, 'Path')[0].split(';'))
except Exception as e: print(e)

try:
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')
    paths.extend(winreg.QueryValueEx(key, 'Path')[0].split(';'))
except Exception as e: print(e)

with open("pythons.txt", "w") as f:
    for p in set(paths):
        if "python" in p.lower():
            f.write(p + "\n")
