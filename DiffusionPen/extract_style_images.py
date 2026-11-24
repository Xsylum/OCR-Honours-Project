"""Extract IAM word images from train_word_IAM.pt into iam_data/words/<a01>/<a01-000u>/<a01-000u-00-00>.png."""
import argparse, os, time
import torch

def log(*a):
    print(*a, flush=True)

p = argparse.ArgumentParser()
p.add_argument('--pt', default='saved_iam_data/train_word_IAM.pt')
p.add_argument('--styles', type=int, nargs='+', default=[12], help='style ids (0-338)')
p.add_argument('--all', action='store_true', help='extract every writer')
p.add_argument('--out', default='iam_data/words')
args = p.parse_args()

log(f'loading {args.pt} ({os.path.getsize(args.pt) / 1e9:.2f} GB) ... this is the slow part')
t = time.time()
data = torch.load(args.pt, weights_only=False)    # pickle of PIL images; only load files you trust
log(f'loaded {len(data)} items in {time.time() - t:.0f}s')

wanted = None if args.all else set(args.styles)
saved = 0
for i, (img, text, wid, path) in enumerate(data):
    if i % 5000 == 0:
        log(f'  {i}/{len(data)}  saved {saved}')
    if wanted is None or int(wid) in wanted:
        rel = '/'.join(path.replace('\\', '/').split('/')[-3:])
        out = os.path.join(args.out, rel + '.png')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img.save(out)
        saved += 1
log(f'done: saved {saved} new images into {args.out}')