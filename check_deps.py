import os
import ast
import sys
import pkgutil

def get_external_imports(root_dir):
    external_deps = set()
    local_modules = set()
    
    # 1. Identify local modules (to avoid suggesting you pip install your own code)
    for root, dirs, files in os.walk(root_dir):
        if '.venv' in root or '.git' in root: continue
        for file in files:
            if file.endswith('.py'):
                local_modules.add(file[:-3])
                if root != root_dir:
                    local_modules.add(os.path.basename(root))

    # 2. Identify standard library modules
    # In Python 3.10+, sys.stdlib_module_names is the most accurate
    std_libs = getattr(sys, 'stdlib_module_names', set())
    if not std_libs: # Fallback for older versions
        std_libs = set(m.name for m in pkgutil.iter_modules() if m.module_finder is None)

    # 3. Scan all .py files for imports
    for root, dirs, files in os.walk(root_dir):
        if '.venv' in root or '.git' in root: continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                external_deps.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.level == 0 and node.module:
                                external_deps.add(node.module.split('.')[0])
                except Exception as e:
                    pass # Skip files with syntax errors

    # 4. Filter list
    to_install = []
    # Mapping for packages where import name != pip install name
    mapping = {
        "google": "google-genai",
        "lancedb": "lancedb",
        "rank_bm25": "rank-bm25",
        "dotenv": "python-dotenv",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
        "yaml": "PyYAML",
        "cv2": "opencv-python"
    }

    for dep in sorted(external_deps):
        if dep not in std_libs and dep not in local_modules and dep != 'check_deps':
            install_name = mapping.get(dep, dep.replace('_', '-'))
            to_install.append(install_name)
    
    return to_install

if __name__ == "__main__":
    print("🔍 Scanning project for dependencies...")
    deps = get_external_imports(os.getcwd())
    
    if deps:
        print("\n🚀 Run the following command in your (.venv_gpu) terminal:\n")
        print(f"pip install {' '.join(deps)}")
    else:
        print("\n✅ No external dependencies found.")
