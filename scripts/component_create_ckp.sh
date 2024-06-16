# Baseline

# python pretrain.py wandb=none name=baseline_ckp budget=24 seed=32 train.steps=5000 train.save_init=True impl.print_loss_every_nth_step=10 impl.save_intermediate_checkpoints=True impl.save_every_nth_step=1000

# Pos tagging
python pretrain.py wandb=none name=pos_2layer_ckp budget=22 seed=32 combine=probing combine.components=[pos_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 train.save_init=True impl.print_loss_every_nth_step=10 impl.save_intermediate_checkpoints=True impl.save_every_nth_step=1000
python pretrain.py wandb=none name=pos_16layer_ckp budget=22 seed=32 combine=probing combine.components=[pos_untrain_16.safetensor] combine.per_component_layer_num=16 combine.active=True combine.load_embeddings=True train.steps=5000 train.save_init=True impl.print_loss_every_nth_step=10 impl.save_intermediate_checkpoints=True impl.save_every_nth_step=1000

# NER
python pretrain.py wandb=none name=ner_2layer_ckp budget=22 seed=32 combine=probing combine.components=[ner_untrain_2.safetensor] combine.per_component_layer_num=2 combine.active=True combine.load_embeddings=True train.steps=5000 train.save_init=True impl.print_loss_every_nth_step=10 impl.save_intermediate_checkpoints=True impl.save_every_nth_step=1000
python pretrain.py wandb=none name=ner_16layer_ckp budget=22 seed=32 combine=probing combine.components=[ner_untrain_16.safetensor] combine.per_component_layer_num=16 combine.active=True combine.load_embeddings=True train.steps=5000 train.save_init=True impl.print_loss_every_nth_step=10 impl.save_intermediate_checkpoints=True impl.save_every_nth_step=1000