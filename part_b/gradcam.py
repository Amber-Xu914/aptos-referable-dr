"""4. Grad-CAM saliency maps for a trained Part B model.

    python part_b/gradcam.py --run efficientnet_b0_finetune                 # validation examples
    python part_b/gradcam.py --run efficientnet_b0_finetune --split test    # only after final_test.py

Picks, at the validation-chosen threshold (sens90), the most confident true positives and
true negatives and the worst false positives and false negatives, and draws each image next
to its Grad-CAM overlay. Also reports, over all images of the split, the share of Grad-CAM
mass inside the field of view and inside the central optic-disc-sized region, as a simple
sanity check that the network is not looking at the black border or camera artefacts.

Grad-CAM (Selvaraju et al., ICCV 2017): weight each channel of the last convolutional
feature map by the spatial mean of the gradient of the referable logit, sum, ReLU, upsample.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import add_path_args, get_paths, load_json, load_splits, normalize, read_rgb, save_json
from part_b.hardware import device


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        """x: (1, 3, H, W) normalised tensor -> (cam HxW in [0, 1], P(referable))."""
        import torch
        self.model.eval()
        x = x.clone().requires_grad_(True)       # needed when the backbone is frozen
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0].detach().float().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, float(torch.sigmoid(logit.detach().float())[0])


def overlay(img, cam, alpha=0.45):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def pick_cases(p, thr, n):
    p = p.assign(pred=(p.model_score >= thr).astype(int), y=(p.true_grade >= 2).astype(int))
    tp = p[(p.y == 1) & (p.pred == 1)].nlargest(n, "model_score")
    tn = p[(p.y == 0) & (p.pred == 0)].nsmallest(n, "model_score")
    fp = p[(p.y == 0) & (p.pred == 1)].nlargest(n, "model_score")
    fn = p[(p.y == 1) & (p.pred == 0)].nsmallest(n, "model_score")
    return [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from part_b.models import build_model

    paths = get_paths(a.root, a.cache)
    dev = device()
    summ = load_json(paths.results / f"summary_{a.run}.json")
    ck = torch.load(paths.ckpt / f"{a.run}.pt", map_location=dev)
    model = build_model(ck["model"], pretrained=False).to(dev)
    model.load_state_dict(ck["state"])
    for prm in model.parameters():
        prm.requires_grad_(True)
    cam_fn = GradCAM(model, model.gradcam_layer())
    size = ck["size"]

    preds = pd.read_csv(paths.results / f"preds_{summ['model_type']}.csv")
    preds = preds[preds.split == a.split]
    if preds.empty:
        sys.exit(f"no {a.split} predictions; run final_test.py first for the test split")
    thr = summ["thresholds"]["sens90"]
    groups = pick_cases(preds, thr, a.n)

    def cam_for(id_code):
        img = read_rgb(paths.cache / f"{id_code}.png", size)
        x = torch.from_numpy(normalize(img))[None].to(dev)
        cam, prob = cam_fn(x)
        return img, cam, prob

    # Figure: one row per category, (image, overlay) pairs
    cols = 2 * a.n
    fig, axes = plt.subplots(4, cols, figsize=(2.2 * cols, 9.4))
    axes = np.atleast_2d(axes)
    for r, (label, sub) in enumerate(groups):
        for c in range(a.n):
            ax_img, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
            for ax in (ax_img, ax_cam):
                ax.axis("off")
            if c >= len(sub):
                continue
            row = sub.iloc[c]
            img, cam, prob = cam_for(row.image_id)
            ax_img.imshow(img); ax_cam.imshow(overlay(img, cam))
            ax_img.set_title(f"{label}  grade {row.true_grade}", fontsize=8)
            ax_cam.set_title(f"p={prob:.2f}  {row.image_id[:8]}", fontsize=8)
    fig.suptitle(f"Grad-CAM: {a.run} ({a.split}); threshold {thr:.2f} chosen on validation", fontsize=10)
    fig.tight_layout()
    out = paths.results / f"gradcam_{a.run}_{a.split}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Simple quantitative check over the whole split
    stats = []
    yy, xx = np.mgrid[:size, :size]
    centre = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (0.15 * size) ** 2
    for id_code in preds.image_id.head(a.max_stats):
        img, cam, _ = cam_for(id_code)
        fov = img.max(axis=2) > 10
        tot = cam.sum() + 1e-8
        stats.append(dict(image_id=id_code, cam_in_fov=cam[fov].sum() / tot, cam_centre=cam[centre].sum() / tot))
    st = pd.DataFrame(stats)
    st.to_csv(paths.results / f"gradcam_stats_{a.run}_{a.split}.csv", index=False)
    summ.setdefault("gradcam", {})[a.split] = dict(
        mean_cam_in_fov=float(st.cam_in_fov.mean()), mean_cam_centre=float(st.cam_centre.mean()), n=len(st))
    save_json(summ, paths.results / f"summary_{a.run}.json")
    print(f"mean Grad-CAM share inside field of view: {st.cam_in_fov.mean():.3f}; "
          f"in central 15 % radius: {st.cam_centre.mean():.3f} (n={len(st)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n", type=int, default=3, help="examples per category")
    p.add_argument("--max-stats", type=int, default=10_000, help="images used for the quantitative check")
    main(p.parse_args())
