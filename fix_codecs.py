import os
import glob

pipeline_dir = r'c:\Users\ASUS\OneDrive\Desktop\camera_Intelligence_System_develop\pipelines'
files = glob.glob(os.path.join(pipeline_dir, '*.py'))

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '.webm' in content or 'vp80' in content:
        new_content = content.replace('.webm', '.mp4').replace('vp80', 'mp4v')
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Updated {f}")
