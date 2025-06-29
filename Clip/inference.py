import os
import config
from tqdm import tqdm
from data_utils import make_train_valid_dfs, build_loaders
from model import CLIPModel
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import cv2

def get_image_embeddings(valid_df, model_path, save_path=None):
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(config.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)

    all_embeddings = torch.cat(valid_image_embeddings)
    if save_path:
        torch.save(all_embeddings, save_path)
    return model, all_embeddings


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(config.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{config.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    _, valid_df = make_train_valid_dfs()
    embedding_path = "valid_image_embeddings.pt"
    model_path = "best.pt"

    # 모델 준비
    model = CLIPModel().to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 이미지 임베딩 불러오거나 새로 계산
    if os.path.exists(embedding_path):
        print(f"Loading image embeddings from {embedding_path}")
        image_embeddings = torch.load(embedding_path).to(config.device)
    else:
        print(f"No saved embeddings found. Generating...")
        _, image_embeddings = get_image_embeddings(valid_df, model_path, save_path=embedding_path)

    find_matches(model,
                 image_embeddings,
                 query="mens fishing on the sea",
                 image_filenames=valid_df['image'].values,
                 n=9)
