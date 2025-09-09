import os
import pandas as pd
import json
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 定义数据集路径
MIMII_PATH = 'dataset/MIMII'
URBAN_SOUND_PATH = 'dataset/UrbanSound8K'
ESC_50_PATH = 'dataset/ESC-50-master'
VGGSOUND_PATH = 'dataset/vggdata'


# 读取UrbanSound8K和ESC-50元数据
urban_sound_metadata = pd.read_csv(os.path.join(URBAN_SOUND_PATH, 'UrbanSound8K.csv'))
esc_50_metadata = pd.read_csv(os.path.join(ESC_50_PATH, 'meta', 'esc50.csv'))

def get_mimii_classes():
    # MIMII数据集的类别是通过文件夹名称定义的
    machine_types = ['fan', 'pump', 'slider', 'valve', 'ToyCar']
    classes = []
    for machine in machine_types:
        classes.extend([f"{machine}_normal", f"{machine}_abnormal"])
    return classes

def get_urbansound_classes():
    # 从UrbanSound8K的metadata文件中读取类别
    urban_sound_metadata = pd.read_csv(os.path.join(URBAN_SOUND_PATH, 'UrbanSound8K.csv'))
    return sorted(urban_sound_metadata['class'].unique().tolist())

def get_esc50_classes():
    # 从ESC-50的metadata文件中读取类别
    esc_50_metadata = pd.read_csv(os.path.join(ESC_50_PATH, 'meta', 'esc50.csv'))
    return sorted(esc_50_metadata['category'].unique().tolist())

def get_vggsound_classes():
    # 从VGGSound的metadata文件中读取类别
    vggsound_metadata = pd.read_csv(
        os.path.join(VGGSOUND_PATH, 'vggsound.csv'),
        header=None,
        names=['youtube_id', 'start_seconds', 'label', 'positive_labels']  # 根据实际列名调整
    )
    return sorted(vggsound_metadata['label'].unique().tolist())

def remove_duplicate_classes(classes_list):
    """移除列表中完全相同的类别名称（不区分大小写）"""
    seen = set()
    unique_classes = []
    for cls in classes_list:
        cls_lower = cls.lower()
        if cls_lower not in seen:
            seen.add(cls_lower)
            unique_classes.append(cls)
    return unique_classes


class SemanticNode:
    def __init__(self, name, children=None, is_leaf=False):
        self.name = name
        self.children = children if children else {}
        self.is_leaf = is_leaf

def remove_underscores(text):
    """将下划线替换为空格"""
    return text.replace('_', ' ')

def calculate_text_similarity(text1, text2, method='tfidf'):
    """计算两个文本之间的相似度"""
    # 在比较之前移除下划线
    text1 = remove_underscores(text1)
    text2 = remove_underscores(text2)
    if method == 'sequence':
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def find_best_category(category, tree):
    """找到语义树中最匹配的类别组"""
    def get_all_categories(node):
        categories = []
        if isinstance(node, dict):
            for key, value in node.items():
                categories.append(key)
                categories.extend(get_all_categories(value))
        return categories

    all_categories = get_all_categories(tree)
    similarities = [(cat, calculate_text_similarity(category, cat)) for cat in all_categories]
    best_match = max(similarities, key=lambda x: x[1])
    return best_match if best_match[1] > 0.3 else None

def insert_category(tree, category, parent_category):
    """将类别插入到语义树中适当的位置"""
    def insert_recursive(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == parent_category:
                    if isinstance(value, dict):
                        if 'other' not in value:
                            value['other'] = []
                        if isinstance(value['other'], list):
                            value['other'].append(category)
                        return True
                    elif isinstance(value, list):
                        value.append(category)
                        return True
                elif insert_recursive(value):
                    return True
        return False
    
    insert_recursive(tree)

def build_semantic_tree():
    """构建音频类别的语义树"""
    semantic_tree = {
        'mechanical_sounds': {
            'industrial_machinery': {
                'fan': ['fan_normal', 'fan_abnormal'],
                'pump': ['pump_normal', 'pump_abnormal'],
                'slider': ['slider_normal', 'slider_abnormal'],
                'valve': ['valve_normal', 'valve_abnormal']
            },
            'vehicles': {
                'car': {
                    'toy_car': ['ToyCar_normal', 'ToyCar_abnormal'],
                    'real_car': ['car_horn', 'engine_idling', 'car_passing']
                },
                'aircraft': ['airplane', 'helicopter'],
                'train': ['train']
            },
            'construction': {
                'tools': ['drilling', 'chainsaw', 'hand_saw', 'jackhammer'],
                'machinery': ['engine']
            },
            'other_mechanical': ['clock_tick', 'clock_alarm']
        },
        'natural_sounds': {
            'animal_sounds': {
                'mammals': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in ['dog', 'cat', 'cow', 'pig', 'sheep', 'mouse'])
                ],
                'birds': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in ['bird', 'crow', 'hen', 'rooster'])
                ],
                'insects': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in ['insects', 'cricket', 'frog'])
                ]
            },
            'weather': {
                'precipitation': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in ['rain', 'snow', 'thunderstorm'])
                ],
                'wind': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in ['wind', 'rustling_leaves'])
                ],
                'thunder': [
                    cls for cls in get_esc50_classes() 
                    if 'thunder' in cls.lower()
                ]
            },
            'water': [
                cls for cls in get_esc50_classes() 
                if any(keyword in cls.lower() for keyword in ['water', 'sea', 'waves', 'stream'])
            ],
            'fire': ['crackling_fire']
        },
        'human_sounds': {
            'voice': {
                'speech': [
                    cls for cls in get_vggsound_classes() 
                    if any(keyword in cls.lower() for keyword in ['speaking', 'talking'])
                ],
                'emotional': [
                    cls for cls in get_vggsound_classes() 
                    if any(keyword in cls.lower() for keyword in ['laughing', 'crying', 'screaming'])
                ],
                'physiological': [
                    cls for cls in get_vggsound_classes() 
                    if any(keyword in cls.lower() for keyword in ['coughing', 'sneezing', 'snoring'])
                ],
                'musical': [
                    cls for cls in get_vggsound_classes() 
                    if any(keyword in cls.lower() for keyword in ['singing', 'whistling', 'humming'])
                ]
            },
            'activities': {
                'movement': ['footsteps', 'running'],
                'hands': ['clapping', 'finger_snapping', 'writing'],
                'daily_activities': ['breathing', 'drinking', 'eating', 'brushing_teeth']
            }
        },
        'urban_sounds': {
            'city_life': {
                'traffic': ['car_horn', 'engine_idling', 'siren', 'car_passing', 'air_horn', 'reversing_beeps'],
                'construction': ['drilling', 'jackhammer'],
                'street_sounds': ['street_music', 'children_playing', 'dog_bark']
            },
            'indoor': {
                'appliances': [
                    cls for cls in get_esc50_classes() 
                    if any(keyword in cls.lower() for keyword in [
                        'washing', 'vacuum', 'clock', 'telephone', 'keyboard_typing',
                        'microwave', 'printer', 'doorbell', 'can_opening'
                    ])
                ],
                'infrastructure': ['door_wood_creaks', 'door_wood_knock', 'glass_breaking', 'elevator']
            },
            'recreational': {
                'entertainment': ['fireworks', 'gunshot'],
                'sports': ['ball_hitting', 'skateboarding']
            }
        }
    }
    
    # 处理未匹配的类别
    unmatched = validate_all_categories(semantic_tree)
    for dataset, categories in unmatched.items():
        for category in categories:
            best_match = find_best_category(category, semantic_tree)
            if best_match:
                insert_category(semantic_tree, category, best_match[0])
    
    return semantic_tree

def print_semantic_tree(tree, level=0):
    """打印语义树结构"""
    for key, value in tree.items():
        print('  ' * level + f'- {remove_underscores(key)}')
        if isinstance(value, dict):
            print_semantic_tree(value, level + 1)
        else:
            for item in value:
                print('  ' * (level + 1) + f'- {remove_underscores(item)}')

def save_semantic_tree(tree, filename='semantic_tree.json'):
    """保存语义树到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

def load_semantic_tree(filename='semantic_tree.json'):
    """从JSON文件加载语义树"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_category_in_tree(category, tree):
    """在语义树中查找类别，返回找到的路径"""
    def search(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = path + [key]
                result = search(value, new_path)
                if result:
                    return result
        elif isinstance(node, list):
            if category in node:
                return path
        return None
    
    return search(tree, [])

def validate_all_categories(tree):
    """验证所有类别是否都能在语义树中找到对应位置"""
    all_categories = {
        'MIMII': get_mimii_classes(),
        'UrbanSound8K': get_urbansound_classes(),
        'ESC-50': get_esc50_classes(),
        'VGGSound': get_vggsound_classes()
    }
    
    unmatched_categories = {}
    
    for dataset, categories in all_categories.items():
        unmatched = []
        for category in categories:
            path = find_category_in_tree(category, tree)
            if not path:
                unmatched.append(category)
        if unmatched:
            unmatched_categories[dataset] = unmatched
    
    return unmatched_categories

if __name__ == "__main__":
    # 获取所有类别并保存到文件
    with open('all_audio_classesv2.txt', 'w', encoding='utf-8') as f:
        # 获取各数据集的原始类别（未去重）
        mimii_classes = get_mimii_classes()
        print(len(mimii_classes))
        # urbansound_classes = get_urbansound_classes()
        # print(len(urbansound_classes))
        esc50_classes = get_esc50_classes()
        print(len(esc50_classes))
#       vggsound_classes = get_vggsound_classes()
        
        # 合并所有类别并去重
        all_classes = mimii_classes  + esc50_classes
        unique_classes = remove_duplicate_classes(all_classes)  # 对所有类别统一去重
        
        # 写入去重后的总类别
        f.write(f"\nTotal unique classes: {len(unique_classes)}\n")
        f.write("\nUnique classes:\n")
        for cls in unique_classes:
            f.write(f"- {remove_underscores(cls)}\n")
    
    print("All audio classes have been saved to 'all_audio_classes.txt'")
