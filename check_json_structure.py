# check_json_structure.py
import json
import os
from pathlib import Path

def check_json_format(json_path):
    """JSON 라벨링 구조 확인"""
    print("="*60)
    print(f"Checking: {os.path.basename(json_path)}")
    print("="*60)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📋 Top-level keys: {list(data.keys())}")
    
    # 전체 구조 출력
    def print_structure(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key, value in list(obj.items())[:5]:  # 처음 5개만
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}: {type(value).__name__}")
                    print_structure(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {value}")
            if len(obj) > 5:
                print(f"{prefix}... and {len(obj)-5} more items")
        elif isinstance(obj, list):
            print(f"{prefix}List length: {len(obj)}")
            if len(obj) > 0:
                print(f"{prefix}First item:")
                print_structure(obj[0], indent + 1)
    
    print("\n📄 Structure:")
    print_structure(data)
    
    # 이미지 정보
    if 'images' in data:
        print(f"\n🖼️ Images: {len(data['images'])}")
        if data['images']:
            print(f"Sample image: {data['images'][0]}")
    
    # Annotation 정보
    if 'annotations' in data:
        print(f"\n📍 Annotations: {len(data['annotations'])}")
        if data['annotations']:
            print(f"Sample annotation:")
            ann = data['annotations'][0]
            for key, value in ann.items():
                if key == 'segmentation' and isinstance(value, list):
                    print(f"  {key}: {len(value)} polygons")
                    if value and isinstance(value[0], list):
                        print(f"    First polygon: {len(value[0])} points")
                else:
                    print(f"  {key}: {value}")
    
    # 카테고리 정보
    if 'categories' in data:
        print(f"\n🏷️ Categories: {len(data['categories'])}")
        for cat in data['categories']:
            print(f"  {cat}")
    
    return data

# 여러 샘플 확인
base_dir = "/Users/admin/Downloads/datasets/1.Training/2.라벨링데이터"

print("Checking COLON samples...")
for class_name in ['궤양', '암', '종양']:
    class_dir = os.path.join(base_dir, '대장', class_name)
    if os.path.exists(class_dir):
        json_files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        if json_files:
            json_path = os.path.join(class_dir, json_files[0])
            check_json_format(json_path)
            print("\n")

print("\n" + "="*60)
print("Checking STOMACH samples...")
print("="*60)
for class_name in ['궤양', '암', '종양']:
    class_dir = os.path.join(base_dir, '위', class_name)
    if os.path.exists(class_dir):
        json_files = [f for f in os.listdir(class_dir) if f.endswith('.json')]
        if json_files:
            json_path = os.path.join(class_dir, json_files[0])
            check_json_format(json_path)
            print("\n")