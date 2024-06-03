"""#исправление слов
!!sudo apt-get install swig3.0
!sudo pip install jamspell
!wget https://github.com/bakwc/JamSpell-models/raw/master/ru.tar.gz
!tar -xvf ru.tar.gz
import jamspell
jsp = jamspell.TSpellCorrector()
assert jsp.LoadLangModel('ru_small.bin')"""

#html разметка
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import pandas as pd
import re
import numpy as np
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("Alexander-896/my-finetuned-bert1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.cpu()

def check_email(i,text):
  j=i
  text+=" "
  sent=""
  while text[j]!=' ':
    sent+=(text[j])
    j+=1
  for x in ['.ru', '.com']:
    if x in sent: return True
  return False

def check_cut_words(i,text,cut_max_len):
  j=i
  sent=""
  while text[j]!=' ' and j>=0:
    sent+=(text[j])
    j-=1
  if len(sent)<=cut_max_len: return True
  return False

def remove_dots(a,b):
  for x in b:
    try:
      a.remove(x)
    except: continue
  return a

def split_text_(ends_of_sents,text):
  ends_of_sents=[0]+list(map(lambda x: x+1,ends_of_sents))+[len(text)]
  sentances=[]
  for i in range(len(ends_of_sents)-1):
      sentances.append(text[ends_of_sents[i]:ends_of_sents[i+1]])
  return sentances

def rule_split_text(text):
  text=text.replace('\n','')
  sent_idx=[]
  bad_sent_idx=[]
  for i in range(len(text)-1):
    if text[i+1] in '?!:;.': sent_idx.append(i+1) #1
    if (text[i] in list(map(str,range(0,10)))) and text[i+1]=='.': bad_sent_idx.append(i+1) #!1
    if text[i+1]=='.' and check_email(i,text): bad_sent_idx.append(i+1) #!2
    if text[i+1]=='.' and check_cut_words(i,text,cut_max_len=3): bad_sent_idx.append(i+1) #!3
  ends_of_sents = remove_dots(sent_idx,bad_sent_idx)
  sentances=split_text_(ends_of_sents,text)
  return sentances


def process_text(text):
    sentances=rule_split_text(text)
    df = pd.DataFrame(sentances, columns=['Text'])

    return df

def encode(docs, tokenizer):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=128, padding='max_length',
                            return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


def prepare_data(input_text,tokenizer):
  text = process_text(input_text)
  input_ids, att_masks = encode(text['Text'].values.tolist(),tokenizer)
  y = torch.LongTensor([0 for x in range(len(text['Text'].values.tolist()))])

  dataset = TensorDataset(input_ids, att_masks, y)
  sampler = RandomSampler(dataset)
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

  return [input_ids, att_masks, y, dataloader]


def get_predict(input_text, model,tokenizer=tokenizer):
  input_ids, att_mask, labels, dataloader=prepare_data(input_text=input_text,tokenizer=tokenizer)
  model.eval()
  pred = []

  with torch.no_grad():
    for step_num, batch_data in tqdm(enumerate(dataloader)):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)


        pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
  pred = np.concatenate(pred)
  return pred


labels_dict={0:"p",1:"h1",2:"em",3:"strong",4:"li"}

def correct_tag(i,text,tag):
  if ';' in text[i]: tag='li'
  if (i-1>=0) and (':' in text[i-1]): tag='li'
  #if ':' in text[i]: tag='strong'
  return tag

def predict_to_html(text_user, predict):
  text = process_text(text_user)['Text'].values.tolist()
  html_text=""
  for i in range(len(predict)):
    tag=labels_dict[predict[i]]
    tag=correct_tag(i,text,tag)
    html_text+=f'<{tag}>{text[i]}</{tag}>'
  return html_text

def correct_html(html_text):
  html_text=html_text.replace('</p><p>',"")
  return html_text

def analizate_text(text):
  text=text.replace('\n',"")
  #text= jsp.FixFragment(text)
  #text = process_text(text)
  #return text
  answer=get_predict(text, model)
  text=predict_to_html(text, answer)
  text=correct_html(text)
  return text

