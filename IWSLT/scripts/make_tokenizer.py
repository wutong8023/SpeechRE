from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("/data/wangguitao/IWSLT/Data/bartbase")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# tokenizer.add_special_tokens({"additional_special_tokens": ["<triplet>", "<subj>", "<obj>"]})
# tokenizer.save_pretrained("/data/wangguitao/IWSLT/Data/bartbase")
print(tokenizer)

"""
PreTrainedTokenizer(name_or_path='/data/wangguitao/IWSLT/Data/bartbase', 
vocab_size=51201, model_max_len=1024, 
is_fast=False, padding_side='right', special_tokens={
'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'sep_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'cls_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True), 
'additional_special_tokens': ['<triplet>', '<subj>', '<obj>']})
"""

"""
PreTrainedTokenizer(name_or_path='/data/wangguitao/IWSLT/Data/bartbase', 
vocab_size=51201, model_max_len=1000000000000000019884624838656, 
is_fast=False, padding_side='right', special_tokens={
'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'sep_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'cls_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True), 
'additional_special_tokens': ['<triplet>', '<subj>', '<obj>']})
"""