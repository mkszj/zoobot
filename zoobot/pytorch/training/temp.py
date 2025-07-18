import torch
import timm
from timm.optim import create_optimizer_v2, param_groups_layer_decay
from timm.scheduler import create_scheduler_v2

# ViT model has (with embed_dim=1152)
# pos_embed [1, 256, 1152]
# patch_embed (16x16 grid for 224x224 image) with patch_embed.proj.weight [3, 1152, 14, 14] and biases [1152]
# 5 blocks.
# each block has qkv attention 16 heads, head dim 72, embed dim 1152, out_features 3456
# and two-layer mlp with 1152->4304->1152

# patch_embed is the first block in my version, and pos_embed is not trained
# then the whole first block gets the same learning rate (both attention and mlp, 6.46e-7)
# then repeat for the remaining identical blocks, with the final 27 block being the whole block at 1e-5

# 0 to 27 = 28 param groups
# 27 after patch (no pos), each one is a block
# example block:
# [torch.Size([1152]), torch.Size([1152]), torch.Size([3456, 1152]), torch.Size([3456]), torch.Size([1152, 1152]), torch.Size([1152]), torch.Size([1152]), torch.Size([1152]), torch.Size([4304, 1152]), torch.Size([4304]), torch.Size([1152, 4304]), torch.Size([1152])]

# pos_embed is the 0 block in timm version (I think), followed by patch_embed (1) with the same lr
# then block 0 gets a slightly higher lr, etc, through the blocks
# each block is enumerated as two param_groups(attention and mlp) so we have pos+patch+(2*27) = 56 param groups
# learning rate is the same for both param groups in a block
# then repeat for all blocks

# 0 to 55 = 56 param groups
# 54 after pos and patch
# so 27 blocks each with 2 param groups, and the first two are pos_embed and patch_embed
# example block
# [torch.Size([1152]), torch.Size([1152]), torch.Size([3456]), torch.Size([1152]), torch.Size([1152]), torch.Size([1152]), torch.Size([4304]), torch.Size([1152]), torch.Size([1152]), torch.Size([1152])] 
# [torch.Size([3456, 1152]), torch.Size([1152, 1152]), torch.Size([4304, 1152]), torch.Size([1152, 4304])] 
# order is a little different but that's not important - same dimensions, I think

def old_style():

    layer_decay = 0.9
    lr = 1e-5
    encoder = timm.create_model('vit_so400m_patch14_siglip_gap_224.v2_webli', pretrained=False)

    params = []

    

    tuneable_blocks = [encoder.patch_embed] + [stage for stage in encoder.blocks]
    tuneable_blocks.reverse()
    blocks_to_tune = tuneable_blocks
    for i, block in enumerate(blocks_to_tune):  # _ is the block name e.g. '3'
        # print(f"Adding block {block} with lr {lr * (layer_decay**i)}")
        params.append({"params": block.parameters(), "lr": lr * (layer_decay**i)})
    params.reverse()  # back to intuitive order
    print('old')
    for i in range(len(params)):
        print(i, params[i]['lr'], [p.size() for p in params[i]['params']], '\n')  # 27 blocks not 50-odd

def new_style():
    encoder = timm.create_model('vit_so400m_patch14_siglip_gap_224.v2_webli', pretrained=False)
    encoder.pos_embed.requires_grad_(False)  # don't train pos_embed
    optimizer = create_optimizer_v2(encoder, 'adamw', lr=1e-5, layer_decay=0.9)
    # print(len(optimizer.param_groups))

    # del optimizer.param_groups[0]

    
    print('new, after removing pos_embed')
    for i in range(len(optimizer.param_groups)):
        print(i,  optimizer.param_groups[i]['lr_scale'] * optimizer.param_groups[0]['lr'], [p.size() for p in optimizer.param_groups[i]['params']], '\n')

    # in my version, pos embed should not be trained, remove



def debug():

    # encoder = timm.create_encoder('vit_small_patch16_224', pretrained=True)
    encoder = timm.create_model('convnext_nano.in12k', pretrained=False)
    # print(encoder)  # Print the encoder architecture

    head = torch.nn.Linear(encoder.num_features, 1000)  # Example head for classification

    # param_groups = param_groups_layer_decay(encoder)
    # print(param_groups) 
    # for group in param_groups:
    #     print(group['lr_scale'], group['weight_decay'], [p.size() for p in param_groups[3]['params']])

    optimizer = create_optimizer_v2(encoder, 'adamw', lr=0.001, weight_decay=0.01)
    print(len(optimizer.param_groups))

    # add head parameters to optimizer
    optimizer.add_param_group({'params': head.parameters(), 'lr': 0.001})
    print(len(optimizer.param_groups))

    print(optimizer)  # Print the optimizer configuration

    scheduler = create_scheduler_v2(optimizer, 'cosine', warmup_epochs=5)
    print(scheduler)  # Print the scheduler configuration


if __name__ == "__main__":

    # old_style()
    new_style()