import json
from visual import TSVFile
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64
from PIL import Image
import io

def load_caps(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            data[imgid] = example['caption']
    return data



img_map = {}
all_img_num = 0
with open('../data/imgs.lineidx.new') as fin:
    for line in fin:
        tokens = line.strip().split('\t')
        all_img_num += 1
        img_map[tokens[0]] = int(tokens[1])

img_tsv = TSVFile('../data/imgs.tsv', all_img_num)
def encode_img(idx):
    offset = img_map[idx]
    img = img_tsv[offset][1]
    return img

img_ids = [30079629,30246015,30138878,30351515,30338284,30039452]
for step, id in enumerate(img_ids):
    img = encode_img(str(id))
    img = Image.open(io.BytesIO(base64.b64decode(img)))
    img.save('./imgs/{}.jpg'.format(step))