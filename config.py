import torch

dir_path = r"C:\Users\xjm12\Desktop\demo1"
train_path = f"{dir_path}/train_3k.txt"
dev_path = f"{dir_path}/dev_1k.txt"
test_path = f"{dir_path}/test_1k.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-chinese"
num_labels = 15
max_length = 128
batch_size = 16
lr = 2e-5
max_epochs = 5
patience = 2
seed_list = [42, 66, 99]

label_id = {
    "100": 0,  "101": 1,  "102": 2,  "103": 3,  "104": 4,
    "106": 5,  "107": 6,  "108": 7,  "109": 8,  "110": 9,
    "112": 10, "113": 11, "114": 12, "115": 13, "116": 14
}