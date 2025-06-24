import importlib
import pkgutil
import inspect

# --- ชื่อคลาสที่ต้องการค้นหา ---
CLASS_TO_FIND = "TokenBucketRateLimiter"

# --- แพ็กเกจที่จะค้นหา ---
PACKAGES_TO_SEARCH = ["langchain", "langchain_core", "langchain_community"]

def search():
    print(f"Searching for class '{CLASS_TO_FIND}'...")
    for package_name in PACKAGES_TO_SEARCH:
        try:
            package = importlib.import_module(package_name)
            if not hasattr(package, '__path__'):
                continue
            
            for _, module_name, _ in pkgutil.walk_packages(package.__path__, package_name + '.'):
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, CLASS_TO_FIND):
                        print("\n" + "="*60)
                        print(f"SUCCESS: Found '{CLASS_TO_FIND}'!")
                        print(f"        It is located in the module: '{module_name}'")
                        print(f"==>     Please use this import statement in your code:")
                        print(f"        from {module_name} import {CLASS_TO_FIND}")
                        print("="*60 + "\n")
                        return # หยุดค้นหาเมื่อเจอ
                except Exception:
                    continue
        except ImportError:
            continue
    print(f"\nSearch complete. Could not find '{CLASS_TO_FIND}'.")

if __name__ == "__main__":
    search()