"""torchrun-launched DDP worker for the distributed determinism test.

Each rank opens the same Ferroload dataset with `world_size`/`rank`, so the
deterministic FerroSampler hands every rank a disjoint, reproducible shard —
that sharding is the thing under test. Per step each rank records:

  - its *local* InfoNCE loss (exercises per-rank sample order), and
  - the *global* loss (all-reduce mean — exercises collective determinism).

Rank 0 gathers every rank's local curve and writes one JSON result file.
The Modal wrapper (or any host) runs this via torchrun:

    torchrun --standalone --nproc-per-node=2 train_ddp.py \
        --data-root /vol/ds --steps 60 --batch-size 32 --out /tmp/ddp_result.json

NCCL note: ring all-reduce order is fixed for a given world size/topology, so
the global curve is bitwise reproducible run-to-run on identical hardware
(pin NCCL_ALGO=Ring to keep NCCL's autotuner from picking a different
algorithm between runs).
"""
import argparse
import json
import os

from common import build_model, captions_to_bow, set_determinism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32, help="per-rank batch size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/ddp_result.json")
    a = ap.parse_args()

    set_determinism(a.seed)
    import torch
    import torch.distributed as dist
    import ferroload

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    # every rank seeds identically -> identical init (DDP's rank-0 broadcast
    # would cover this anyway); the loader is where ranks diverge, on purpose
    dl = ferroload.make_loader(
        a.data_root, batch_size=a.batch_size, images=["image"], meta=["caption"],
        resize=(128, 128), out="torch", shuffle=True, seed=a.seed,
        world_size=world, rank=rank, drop_last=True)

    model = build_model().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    local, global_, step, epoch = [], [], 0, 0
    while step < a.steps:
        dl.set_epoch(epoch)
        for batch in dl:
            x = batch["image"].permute(0, 3, 1, 2).float().div_(255).cuda()
            t = captions_to_bow(batch["caption"]).cuda()
            loss = model(x, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            g = loss.detach().clone()
            dist.all_reduce(g, op=dist.ReduceOp.AVG)
            local.append(loss.item())
            global_.append(g.item())
            step += 1
            if rank == 0 and (step % 25 == 0 or step == a.steps):
                print(f"[ddp] step {step}/{a.steps} global loss {global_[-1]:.6f}",
                      flush=True)
            if step >= a.steps:
                break
        epoch += 1

    all_local = [None] * world
    dist.all_gather_object(all_local, local)
    if rank == 0:
        with open(a.out, "w") as f:
            json.dump({"world_size": world, "losses": global_,
                       "per_rank_local": all_local}, f)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
