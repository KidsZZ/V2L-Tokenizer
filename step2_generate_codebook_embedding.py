import numpy as np
import torch
import open_clip as clip
from tqdm import tqdm

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")
print(f"📊 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

## Load CLIP model
print("🔄 Loading CLIP model...")
model, _, preprocess = clip.create_model_and_transforms('ViT-L-14',pretrained='laion2b_s32b_b82k',device=DEVICE,cache_dir="/root/autodl-tmp/downloads")
model.to(DEVICE)
print(f"✅ CLIP model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check distributed training
try:
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        print("🔗 Distributed training enabled")
except NameError:
    print("ℹ️  Distributed training not enabled (args not defined)")

# Load the vocabulary from the numpy file
print("📖 Loading vocabulary...")
llama_texts = np.load("Subword_Bigram_Trigram_Vocabulary.npy", allow_pickle=True)
print(f"📚 Vocabulary loaded, {len(llama_texts)} entries")

# Analyze vocabulary structure
sample_entry = llama_texts[0] if len(llama_texts) > 0 else None
if sample_entry:
    print(f"📋 Vocabulary sample entry: {dict(sample_entry)}")
    print(f"🔑 Vocabulary keys: {list(sample_entry.keys())}")

local_codebook = []
global_codebook = []

####Generate Subword Embeddings
print("\n🔤 Generating subword embeddings...")
print(f"📏 ImageNet templates: {len(imagenet_templates)}")

for i, token_cell in enumerate(tqdm(llama_texts, desc="Processing subwords", unit="word")):
    cur_token = token_cell["1"]
    
    if i % 1000 == 0:
        print(f"📝 Processing {i}th word: '{cur_token}'")
    
    texts = [template.format(cur_token) for template in imagenet_templates] 
    
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    
    global_codebook.append(text_features)
    local_codebook.append(text_features)

print(f"✅ Subword embeddings generated, {len(local_codebook)} vectors")

####Generate Bigrams Embeddings
print("\n🔤🔤 Generating bigram embeddings...")
bigram_count = 0
for i, token_cell in enumerate(tqdm(llama_texts, desc="Processing bigrams", unit="word pair")):
    if "2" not in token_cell:
        continue
        
    cur_token = token_cell["1"] + token_cell["2"]
    bigram_count += 1
    
    if i % 1000 == 0:
        print(f"📝 Processing {bigram_count}th bigram: '{cur_token}'")
    
    texts = [template.format(cur_token) for template in imagenet_templates] 
    
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    global_codebook.append(text_features)

print(f"✅ Bigram embeddings generated, {bigram_count} vectors")

####Generate Trigrams Embeddings
print("\n🔤🔤🔤 Generating trigram embeddings...")
trigram_count = 0
for i, token_cell in enumerate(tqdm(llama_texts, desc="Processing trigrams", unit="word group")):
    if "3" not in token_cell:
        continue
        
    cur_token = token_cell["1"] + token_cell["2"] + token_cell["3"]
    trigram_count += 1
    
    if i % 1000 == 0:
        print(f"📝 Processing {trigram_count}th trigram: '{cur_token}'")
    
    texts = [template.format(cur_token) for template in imagenet_templates] 
    text_features = clip.tokenize(texts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = torch.mean(text_features, dim=0)
    text_features = text_features.unsqueeze(0)
    global_codebook.append(text_features)

print(f"✅ Trigram embeddings generated, {trigram_count} vectors")

##Global Codebook: Both Subword Bigram Trigram
print("\n💾 Saving global codebook...")
global_codebook = torch.concat(global_codebook, dim=0)
print(f"📊 Global codebook shape: {global_codebook.shape}")
print(f"📈 Global codebook stats: mean={global_codebook.mean():.4f}, std={global_codebook.std():.4f}")
torch.save(global_codebook, "Subword_Bigram_Trigram_Embedding.pth")
print("✅ Global codebook saved: Subword_Bigram_Trigram_Embedding.pth")

##Local Codebook: Only for Subwords
print("\n💾 Saving local codebook...")
local_codebook = torch.concat(local_codebook, dim=0)
print(f"📊 Local codebook shape: {local_codebook.shape}")
print(f"📈 Local codebook stats: mean={local_codebook.mean():.4f}, std={local_codebook.std():.4f}")
torch.save(local_codebook, "local_codebook_embedding.pth")
print("✅ Local codebook saved: local_codebook_embedding.pth")

print(f"\n🎉 Codebook generation complete!")
print(f"📋 Summary:")
print(f"   - Subwords: {len(local_codebook)}")
print(f"   - Bigrams: {bigram_count}")
print(f"   - Trigrams: {trigram_count}")
print(f"   - Global codebook size: {global_codebook.shape[0]}")
print(f"   - Embedding dimension: {global_codebook.shape[1]}")