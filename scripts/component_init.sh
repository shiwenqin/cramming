# Baseline
# CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=baseline_shuffle budget=22 seed=32 train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# rc
CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=rc_2layer budget=23 seed=32 combine=probing combine.components=[rc_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True
CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=rc_16layer budget=23 seed=32 combine=probing combine.components=[rc_untrain_16.safetensor] combine.per_component_layer_num=16 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# # NER
# CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=ner_2layer budget=22 seed=32 combine=probing combine.components=[ner_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True
# CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=ner_16layer budget=22 seed=32 combine=probing combine.components=[ner_untrain_16.safetensor] combine.per_component_layer_num=16 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# Pos NER
CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=pos_ner_rc_2layer budget=22 seed=32 combine=probing combine.components=[pos_untrain_2.safetensor,ner_untrain_2.safetensor,rc_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# NER Pos
CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=rc_ner_pos_2layer budget=22 seed=32 combine=probing combine.components=[rc_untrain_2.safetensor,ner_untrain_2.safetensor,pos_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# Pos * 8
CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=rcx8_2layer budget=22 seed=32 combine=probing combine.components=[rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor,rc_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True

# # NER * 8
# CUDA_VISIBLE_DEVICES=7 python pretrain.py wandb.project=component_exp_fix_init name=nerx8_2layer budget=22 seed=32 combine=probing combine.components=[ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor,ner_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 impl.print_loss_every_nth_step=10 impl.shuffle_in_dataloader=True