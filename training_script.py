import tarfile
import shutil
import subprocess
import os

data_dir = '/opt/ml/input/data/training'
source_code_dir = '/opt/ml/code'

for item in os.listdir(data_dir):
    source_item = os.path.join(data_dir, item)
    
    # Chỉ sao chép các tệp (bỏ qua thư mục)
    if os.path.isfile(source_item):
        destination_item = os.path.join(source_code_dir, item)
        
        # Sao chép tệp vào thư mục đích
        shutil.copy2(source_item, destination_item)  # Dùng copy2 để giữ nguyên thời gian sửa đổi tệp
        print(f"Đã sao chép {source_item} -> {destination_item}")

#shutil.copytree(data_dir, source_code_dir, dirs_exist_ok=True)
#shutil.copytree(data_dir, source_code_dir, dirs_exist_ok=True)


source_dir = '/opt/ml/input/data/code'  
destination_dir = '/opt/ml/code'  # Thư mục đích

# Kiểm tra nếu thư mục đích không tồn tại, tạo mới
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Duyệt qua cây thư mục và sao chép các tệp vào đúng thư mục đích
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Đường dẫn đầy đủ đến tệp nguồn
        source_file = os.path.join(root, file)
        
        # Tạo đường dẫn đích sao cho giữ lại cấu trúc thư mục của thư mục con
        relative_path = os.path.relpath(root, source_dir)  # Tính đường dẫn tương đối từ source_dir đến root
        destination_folder = os.path.join(destination_dir, relative_path)  # Tạo thư mục đích tương ứng

        # Kiểm tra xem thư mục đích đã tồn tại chưa, nếu chưa thì tạo nó
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Đường dẫn đích cho tệp
        destination_file = os.path.join(destination_folder, file)

        # Sao chép tệp từ thư mục nguồn vào thư mục đích
        shutil.copy(source_file, destination_file)

        print(f'Successfully copied: {source_file} -> {destination_file}')

with tarfile.open('data_train.tar.gz', 'r:gz') as tar:
    tar.extractall()

subprocess.run([
    'python', 'training.py',
    '--output_dir', './OUTPUT_desenet121/',
    '--log_dir', './LOG_desenet121/',
    '--batch_size', '6',
    '--model', 'desenet121',
    '--mask_ratio', '0.75',
    '--epochs', '5',
    '--warmup_epochs', '2',
    '--blr', '1.5e-4',
    '--weight_decay', '0.05',
    '--num_workers', '5',
    '--input_size', '224',
    '--random_resize_range', '0.5', '1.0',
    '--datasets_names', 'chexpert',
    '--device', 'cpu'
], check=True)

source_dir = './OUTPUT_desenet121/'
destination_dir = '/opt/ml/model/'

os.makedirs(destination_dir, exist_ok=True)

shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
