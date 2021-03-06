# GPT2-Chinese-Couplet
## Description
This is a project I did earlier this year, a transformer-based generation model that is focusing on completing Chinese couplets. This model has integrated controlled text generation with plug and play language model by Uber’s research team.
This project extends from the "GPT2-Chinese" and Uber's "PPLM" projects, please read the "LICENSE_GPT2_Chinese" and "LICENSE_PPLM" first

## Preparation:
pip3 install -r requirements.txt

## Training:
python3 train.py --raw
- adjust hyperparameters accroding train.py if you wish to
- a pre-trained demo is attached. training may take a long time, so you probably don't want to do it again.

## Generating without PPLM:
python3 generate.py --prefix '爆竹声声辞旧岁|'

## Generating with PPLM:
python3 generate.py --prefix '爆竹声声辞旧岁|' --bow --bow_path 'data/bow_newyear.txt'

- adjust hyperparameters accroding generate.py if you wish to
- --prefix sets up the incomplete couplet, '|' is the optional symbol indicate the end of the first line.
- there are plenty of couplets in the "data" folder as well as on the internet that you can try them out, note the couplet must be <=69 characters long
- --bow indicates the PPLM-BOW is enabled.
- --bow_path sets the path of the bag of considered characters.

## File path
- the pre-defined configuration of the model is stored in the "config" folder.
- bow files are stored in the "data" folder.
- vocabulary file is stored in the "data" folder.
- couplets file is stored in the "data" folder.
