# The jsonl profile plus a checkpoint pattern and one more metric.
base = jsonl
ckpt_prefix = ckpt_step
ckpt_suffix = .pt
loss_fields = "loss":, "gnorm":, "tok_s":
loss_labels = loss, gnorm, tok/s
