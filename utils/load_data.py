import os

def load_dataset(main_folder):
    texts = []
    labels = []
    for author_folder in os.listdir(main_folder):
        author_path = os.path.join(main_folder, author_folder)
        if os.path.isdir(author_path):
            for txt_file in os.listdir(author_path):
                file_path = os.path.join(author_path, txt_file)
                if file_path.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                        labels.append(author_folder)
    return texts, labels