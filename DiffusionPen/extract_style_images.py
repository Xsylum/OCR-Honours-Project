"""Extract IAM word images for chosen style (writer) ids from DiffusionPen's train_word_IAM.pt
into ./iam_data/words/<a01>/<a01-000u>/<a01-000u-00-00>.png, the layout train.py's sampling reads."""
import argparse
import os
import torch

p = argparse.ArgumentParser()
p.add_argument('--pt', default='saved_iam_data/train_word_IAM.pt')
p.add_argument('--styles', type=int, nargs='+', default=[12], help='style ids (0-338) to extract')
p.add_argument('--out', default='iam_data/words')
args = p.parse_args()

# The file is a pickle of PIL images; only load files you trust (this one is from the DiffusionPen authors).
data = torch.load(args.pt, weights_only=False)
print('items:', len(data))
print('first item (text, writer id, path):', data[0][1:])

wanted = set(args.styles)
saved = 0
for img, text, wid, path in data:
    if int(wid) in wanted:
        rel = '/'.join(path.replace('\\', '/').split('/')[-3:])   # a01/a01-000u/a01-000u-00-00
        out = os.path.join(args.out, rel + '.png')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img.save(out)
        saved += 1
print(f'saved {saved} images for styles {sorted(wanted)} into {args.out}')
