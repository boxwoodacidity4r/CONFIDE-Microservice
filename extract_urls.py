import os
import re

def extract_urls_from_java_file(file_path):
    urls = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            # Spring MVC
            spring_patterns = [
                r'@RequestMapping\([^)]*["\']([^"\']+)["\']',
                r'@GetMapping\([^)]*["\']([^"\']+)["\']',
                r'@PostMapping\([^)]*["\']([^"\']+)["\']',
                r'@PutMapping\([^)]*["\']([^"\']+)["\']',
                r'@DeleteMapping\([^)]*["\']([^"\']+)["\']'
            ]

            # JAX-RS
            jaxrs_patterns = [
                r'@Path\("([^"]+)"\)',
            ]

            for pattern in spring_patterns + jaxrs_patterns:
                matches = re.findall(pattern, content)
                urls.extend(matches)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return urls

def extract_urls_from_webxml(file_path):
    urls = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            matches = re.findall(r'<url-pattern>(.*?)</url-pattern>', content)
            urls.extend(matches)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return urls

def scan_project(root_dir):
    all_urls = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".java"):
                all_urls.extend(extract_urls_from_java_file(file_path))
            elif file == "web.xml":
                all_urls.extend(extract_urls_from_webxml(file_path))
    return sorted(set(all_urls))

if __name__ == "__main__":
    # 改成你的项目路径
    project_root = r"D:\multimodal_microservice_extraction\data\raw\acmeair-monolithic-java\src\main\java"
    urls = scan_project(project_root)
    print("发现的URL映射：")
    if not urls:
        print("⚠️ 没找到 URL，请检查是否在 web.xml 或 JS 文件中。")
    for url in urls:
        print(url)
