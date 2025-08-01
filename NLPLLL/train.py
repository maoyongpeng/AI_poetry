from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import pickle as p

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.nn.utils import clip_grad_norm_
import tqdm

import data_process
from configs import opt_shi,opt_ci
from model import PoetryModel

from sample import generatePoetry



opt=opt_shi
data,word_to_ix,ix_to_word = data_process.optimize_get_data(opt)
# print(data.dtype)  # np.array

data=torch.tensor(data,dtype=torch.long)

dataloader=DataLoader(TensorDataset(data),batch_size=opt.batch_size,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():


    model = PoetryModel(len(word_to_ix))

    if opt.model_load_path is not None: # 如果有预训练模型
        model.load_state_dict(torch.load(opt.model_load_path,map_location=device))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,betas=(0.9,0.98),eps=1e-9)
    scheduler=ReduceLROnPlateau(optimizer,'min',patience=2,factor=0.5,verbose=True)
    criterion =nn.CrossEntropyLoss(ignore_index=word_to_ix['<PAD>'])


    epochNum = opt.epoch


    print("训练开始!")

    for epoch in range(epochNum):
        loss_calc=0.0
        model.train() # 进入训练模式
        progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochNum}",
                                 ncols=100)
        for ii, (data_,) in progress_bar:
            # 训练
            data_ = data_.transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]

            src_mask = generate_square_subsequent_mask(input_.size(0)).to(device)
            src_pad_mask = (input_ == word_to_ix['<PAD>']).to(device)
            src_pad_mask = src_pad_mask.permute(1, 0).contiguous()

            memory, logit = model(input_, src_mask, src_pad_mask)
            mask = target != word_to_ix['<PAD>']
            target = target[mask]  # 去掉前缀的空格
            logit = logit.flatten(0, 1)[mask.view(-1)]

            loss = criterion(logit, target)
            loss.backward()
            clip_grad_norm_(model.parameters(),max_norm=1.0) # 梯度裁剪
            optimizer.step()
            loss_calc+=loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss=loss_calc/len(dataloader)
        scheduler.step(avg_loss) #调度器更新
        # 将loss写入txt文件
        with open('loss.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}\n")

        model.eval()  # 进入评估模式
        # 将生成的诗写入txt文件,开头的字为“春”,"水","月"
        gen_model = generatePoetry(model, word_to_ix, ix_to_word, max_length=opt.generate_maxlen, device=device)
        with open('generated_poetry.txt', 'a') as f:
            f.write(f"Epoch {epoch}\n")
            f.write(f"春: {gen_model.sample_with_temperature(startWord='春',temperature=0.7)}\n")
            f.write(f"水: {gen_model.sample_with_temperature(startWord='水',temperature=0.7)}\n")
            f.write(f"给句成诗(海内存知己):\n")
            f.write(f"{gen_model.gen(start_words='海内存知己',temperature=0.7)}")
            f.write(f"藏头诗(深度学习):\n")
            f.write(f"{gen_model.gen_acrostic_T(start_words='深度学习',temperature=0.7)}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{opt.model_save_path}_{epoch + 1}.pth")


    print("训练结束!")


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # 生成下三角矩阵（下三角全为True，其余位False）
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
if __name__ == '__main__':
    train()


    pass