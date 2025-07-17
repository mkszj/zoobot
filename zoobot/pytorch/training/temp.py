import torch
import timm
from timm.optim import create_optimizer_v2, param_groups_layer_decay
from timm.scheduler import create_scheduler_v2

if __name__ == "__main__":

    # encoder = timm.create_encoder('vit_small_patch16_224', pretrained=True)
    encoder = timm.create_model('convnext_nano.in12k', pretrained=True)
    # print(encoder)  # Print the encoder architecture

    head = torch.nn.Linear(encoder.num_features, 1000)  # Example head for classification

    # param_groups = param_groups_layer_decay(encoder)
    # print(param_groups) 
    # for group in param_groups:
    #     print(group['lr_scale'], group['weight_decay'], [p.size() for p in param_groups[3]['params']])

    # higher layer_decay value causes SLOWER decay of learning rate
    optimizer = create_optimizer_v2(encoder, 'adamw', lr=0.001, weight_decay=0.01, layer_decay=0.9)
    # print(len(optimizer.param_groups))

    # add head parameters to optimizer
    optimizer.add_param_group({'params': head.parameters(), 'lr': 0.001})
    # print(len(optimizer.param_groups))

    print('before manually applying lr_scale')
    for group in optimizer.param_groups:
        print('Group LR:', group['lr'], 'Group LR Scale:', group.get('lr_scale', None), 'Weight Decay:', group['weight_decay'])

    # lr_scale doesn't actually interact with the optimizer in timm, it's just a metadata field
    # timm scheduler uses lr_scale to do value = value * lr_scale
    # so if you have no scheduler, lr_scale has no effect
    # so you need to manually apply value * lr_scale


    for group in optimizer.param_groups:
        group['lr_scale'] = group.get('lr_scale', 1.0)
        group['lr'] *= group['lr_scale']
    print('after manually applying lr_scale')
    for group in optimizer.param_groups:
        print('Group LR:', group['lr'], 'Group LR Scale:', group.get('lr_scale', None), 'Weight Decay:', group['weight_decay'])


    
    
    

    # print(optimizer)  # Print the optimizer configuration

    # scheduler = create_scheduler_v2(optimizer, 'cosine', warmup_epochs=5)
    # print(scheduler)  # Print the scheduler configuration
