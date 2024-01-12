import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from util_funcs import get_length_param, get_user_param, build_text_file,load_dataset
import sys
import re
import json

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler


def chat_function(message, length_of_the_answer, who_is_next, creativity):   # model, tokenizer

    input_user = message

    if length_of_the_answer == 'short':
        next_len = '1'
    elif length_of_the_answer == 'medium':
        next_len = '2'
    elif length_of_the_answer == 'long':
        next_len = '3'
    else:
        next_len = '-'

    print(who_is_next)
    if who_is_next == 'Zak':
        next_who = 'G'
    elif who_is_next == 'Shiv Bhonde':
        next_who = 'G'
    else: next_who = 'H'



    history = gr.get_state() or []
    chat_history_ids = torch.zeros((1, 0), dtype=torch.int) if history == [] else torch.tensor(history[-1][2], dtype=torch.long)

    # encode the new user input, add parameters and return a tensor in Pytorch
    if len(input_user) != 0:

        new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
                                              + input_user + tokenizer.eos_token, return_tensors="pt")
        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        input_user = '-'

    if next_who == "G":

        # encode the new user input, add parameters and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

        print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input

        # save previous len
        input_len = chat_history_ids.shape[-1]
        # generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
        chat_history_ids = model.generate(
            chat_history_ids,
            num_return_sequences=1,                     # use for more variants, but have to print [i]
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature = float(creativity),                          # 0 for greedy
            mask_token_id=tokenizer.mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            device='cpu'
        )

        response = tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)
    else:
        response = '-'

    history.append((input_user, response, chat_history_ids.tolist()))
    gr.set_state(history)

    html = "<div class='chatbot'>"
    for user_msg, resp_msg, _ in history:
        if user_msg != '-':
            html += f"<div class='user_msg'>{user_msg}</div>"
        if resp_msg != '-':
            html += f"<div class='resp_msg'>{resp_msg}</div>"
    html += "</div>"
    return html





# Download checkpoint:
checkpoint = "cognitivecomputations/dolphin-2.2.1-mistral-7b"
tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# model = model.eval()

#@markdown Your telegram chat json path 'ChatExport.../YourChatName.json':
path_to_telegram_chat_json = 'result.json' #@param {type : "string"}
#@markdown Name of the user to predict by GPT-3:
machine_name_in_chat = 'Zak' #@param {type : "string"}


with open(path_to_telegram_chat_json, encoding='utf-8') as f: data = json.load(f)['messages']

# test data is first 10% of chat, train - last 90%
train, test = data[int(len(data)*0.1):], data[:int(len(data)*0.1)]

build_text_file(train, 'train_dataset.txt', tokenizer)
build_text_file(test,  'test_dataset.txt', tokenizer)

print("Train dataset length: " + str(len(train)) + "samples")
print("Test dataset length: "  + str(len(test)) + "samples")

# Create PyTorch Datasets
train_dataset, test_dataset, data_collator = load_dataset('train_dataset.txt', 'test_dataset.txt', tokenizer)

# Create PyTorch Dataloaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=data_collator)

# this cell checks 1 forward pass
try:
    for batch in train_loader:
        break
    {k: v.shape for k, v in batch.items()}

    outputs = model(**batch)
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

#@title Fine-tuning params
num_epochs = 2 #@param {type:"integer"}
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
save_checkpoint_path = 'GPT3_checkpoint-more-data-2ep.pt' #@param {type:"string"}


num_training_steps = num_epochs * len(train_dataset)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

accelerator = Accelerator()
train_dl, test_dl, model, optimizer = accelerator.prepare(
    train_loader, test_loader, model, optimizer
)
# wandb.watch(model, log="all")

print("Start training...")
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):

    ### TRAIN EPOCH
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        # wandb.log({'train_loss':loss.item()})
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    ### SAVE
    print("Saving model...")
    torch.save({
            'model_state_dict': model.state_dict(),
    }, save_checkpoint_path)

    ### VALIDATE ONCE
    cum_loss = 0
    model.eval()
    with torch.inference_mode():
        for batch in test_dl:
            outputs = model(**batch)
            cum_loss += float(outputs.loss.item())

    print(cum_loss/len(test_loader))
    # wandb.log({'val_mean_loss':cum_loss/len(test_loader)})



# # Gradio
# checkbox_group = gr.inputs.CheckboxGroup(['Zak', 'Me'], default=['Zak'], type="value", label=None)
# title = "Chat with Zak (in Russian)"
# description = "This is where you can chat with me. But there's a bot instead of me. Leave the message blank so Zak can keep talking. Read more about the technique in the link below."
# article = "<p style='text-align: center'><a href='https://github.com/Kirili4ik/ruDialoGpt3-finetune-colab'>Github with fine-tuning GPT-3 on your chat</a></p>"
# examples = [
#             ["Hey, how's it going?", 'medium', 'Zak', 0.5],
#             ["How old are you?", 'medium', 'Zak', 0.3],
# ]

# iface = gr.Interface(chat_function,
#                      [
#                          "text",
#                          gr.inputs.Radio(["short", "medium", "long"], default='medium'),
#                          gr.inputs.Radio(["Zak", "Me"], default='Zak'),
#                          gr.inputs.Slider(0, 1, default=0.5)
#                      ],
#                      "html",
#                      title=title, description=description, article=article, examples=examples,
#                      css= """
#                             .chatbox {display:flex;flex-direction:column}
#                             .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
#                             .user_msg {background-color:cornflowerblue;color:white;align-self:start}
#                             .resp_msg {background-color:lightgray;align-self:self-end}
#                           """,
#                      allow_screenshot=True,
#                      allow_flagging=False
#                     )

# if __name__ == "__main__":
#     iface.launch()
