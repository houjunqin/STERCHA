import os
import time
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import STERCHA
from utils import log_string
from metrics import masked_mae, masked_mape, masked_rmse


class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.cfg_model = {**kwargs.get("data", {}), **kwargs.get("STERCHA", {})}
        self.log_dir = self.cfg_model.get("log", "logs/STERCHA/")
        self.model_file = os.path.join(self.log_dir, "best_model.pth")

        self.input_feature_idx = self.cfg_model.get("input_feature_idx", [0])
        self.output_feature_idx = self.cfg_model.get("output_feature_idx", [0])
        self.input_dim = self.cfg_model.get("input_dim", 10)
        self.output_dim = len(self.output_feature_idx)
        self.horizon = self.cfg_model.get("horizon", 12)

        # mode switches
        self.train_mode = self.cfg_model.get("train_mode", "current")   # "current" | "rollout"
        self.val_mode = self.cfg_model.get("val_mode", "current")       # "current" | "rollout"
        self.test_mode = self.cfg_model.get("test_mode", "rollout")     # "current" | "rollout"
        self.rollout_steps = self.cfg_model.get("rollout_steps", self.horizon)

        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_distributed()
        self.gradscaler = amp.GradScaler(enabled=self.use_amp)
        self.build_model()

    def _setup_distributed(self):
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_amp = self.device.type == "cuda"
        self.is_main_process = self.rank == 0

    def cleanup(self):
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _log(self, msg: str):
        if self.is_main_process:
            log_string(self.log_dir, msg)

    def _reduce_mean(self, value):
        if not self.distributed:
            return value
        tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor.item()

    def build_model(self):
        model = STERCHA(self.cfg_model).to(self.device)

        if self.distributed:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg_model["learning_rate"],
            weight_decay=self.cfg_model.get("weight_decay", 1e-4),
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.cfg_model.get("scheduler_mode", "min"),
            factor=self.cfg_model.get("scheduler_factor", 0.97),
            patience=self.cfg_model.get("scheduler_patience", 3),
            threshold=self.cfg_model.get("scheduler_threshold", 0.1),
        )

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _run_model(self, x, tod, dow, bathymetry, manning, mode: str):
        if mode == "current":
            return self.model(
                x, tod, dow, bathymetry, manning,
                mode="current",
            )
        elif mode == "rollout":
            return self.model(
                x, tod, dow, bathymetry, manning,
                mode="rollout",
                rollout_steps=self.rollout_steps,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def inverse_transform_by_feature_group(self, tensor, feature_idx, scaler, scaler_wave):
        STATE_FEATURES = list(range(0, 6))
        WAVE_FEATURES = list(range(9, 14))
        out = tensor.clone()

        state_pairs = [(local_i, global_i) for local_i, global_i in enumerate(feature_idx) if global_i in STATE_FEATURES]
        if state_pairs:
            full_state = torch.zeros(*tensor.shape[:-1], len(STATE_FEATURES), dtype=tensor.dtype, device=tensor.device)
            for local_i, global_i in state_pairs:
                full_state[..., global_i] = tensor[..., local_i]
            full_state_inv = scaler.inverse_transform(full_state)
            for local_i, global_i in state_pairs:
                out[..., local_i] = full_state_inv[..., global_i]

        wave_pairs = [(local_i, global_i) for local_i, global_i in enumerate(feature_idx) if global_i in WAVE_FEATURES]
        if wave_pairs:
            full_wave = torch.zeros(*tensor.shape[:-1], len(WAVE_FEATURES), dtype=tensor.dtype, device=tensor.device)
            for local_i, global_i in wave_pairs:
                full_wave[..., global_i - 9] = tensor[..., local_i]
            full_wave_inv = scaler_wave.inverse_transform(full_wave)
            for local_i, global_i in wave_pairs:
                out[..., local_i] = full_wave_inv[..., global_i - 9]

        return out

    def train(self):
        self._log("====================== START TRAINING ======================")

        training_history = []
        start_total_time = time.time()
        min_val_mae = float("inf")
        early_stopping = EarlyStopping(patience=self.cfg_model.get("patience", 20))

        # ---------------------------------------------------------
        # Preload static per-subgraph objects once
        # ---------------------------------------------------------
        subgraphs = {}
        for subgraph_idx in range(1, len(self.datasets) + 1):
            sub = self.datasets[subgraph_idx]

            subgraphs[subgraph_idx] = {
                "sub": sub,
                "scaler": sub["scaler"],
                "scaler_wave": sub["scaler_wave"],
                "train_loader": sub["X_train_loader_wave"],
                "val_loader": sub["X_val_loader_wave"],
                "test_loader": sub.get("X_test_loader_wave", None),
                "bathymetry": torch.tensor(
                    np.copy(sub["bathymetry_manning"][:, 0]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "manning": torch.tensor(
                    np.copy(sub["bathymetry_manning"][:, 1]),
                    dtype=torch.float32,
                    device=self.device,
                ),
            }

        # ---------------------------------------------------------
        # Epoch loop: every epoch sees all subgraphs
        # ---------------------------------------------------------
        for epoch in range(1, self.cfg_model["epochs"] + 1):
            epoch_start = time.time()

            # ---------- TRAIN ----------
            self.model.train()

            epoch_train_loss = []
            epoch_train_mape = []
            epoch_train_rmse = []

            for subgraph_idx in range(1, len(self.datasets) + 1):
                item = subgraphs[subgraph_idx]
                train_loader = item["train_loader"]
                scaler = item["scaler"]
                scaler_wave = item["scaler_wave"]
                bathymetry = item["bathymetry"]
                manning = item["manning"]

                train_sampler = getattr(train_loader, "sampler", None)
                if self.distributed and isinstance(train_sampler, DistributedSampler):
                    # make each epoch shuffle correctly per subgraph
                    train_sampler.set_epoch(epoch * 1000 + subgraph_idx)

                pbar = tqdm(
                    train_loader,
                    desc=f"Rank {self.rank} | Epoch {epoch} | Train Sub {subgraph_idx}",
                    leave=False,
                    disable=not self.is_main_process,
                )

                sub_train_loss = []
                sub_train_mape = []
                sub_train_rmse = []

                for x, y, _ in pbar:
                    dow = x[..., -8].to(self.device, non_blocking=True)
                    tod = x[..., -7].to(self.device, non_blocking=True)

                    x = torch.as_tensor(
                        x[..., self.input_feature_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    y = torch.as_tensor(
                        y[..., self.output_feature_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    self.optimizer.zero_grad(set_to_none=True)

                    with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        y_hat = self._run_model(
                            x, tod, dow, bathymetry, manning,
                            mode=self.train_mode,
                        )

                        y_target = y[:, :y_hat.shape[1]]

                        y_hat_phys = self.inverse_transform_by_feature_group(
                            y_hat, self.output_feature_idx, scaler, scaler_wave
                        )
                        y_phys = self.inverse_transform_by_feature_group(
                            y_target, self.output_feature_idx, scaler, scaler_wave
                        )

                        total_loss = masked_mae(y_hat_phys, y_phys, 0.0)

                    self.gradscaler.scale(total_loss).backward()
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()

                    sub_train_loss.append(total_loss.item())
                    sub_train_mape.append(masked_mape(y_hat_phys, y_phys, 0.0).item())
                    sub_train_rmse.append(masked_rmse(y_hat_phys, y_phys, 0.0).item())

                # reduce each subgraph result across ranks
                sub_loss_mean = self._reduce_mean(np.mean(sub_train_loss)) if sub_train_loss else np.nan
                sub_mape_mean = self._reduce_mean(np.mean(sub_train_mape)) if sub_train_mape else np.nan
                sub_rmse_mean = self._reduce_mean(np.mean(sub_train_rmse)) if sub_train_rmse else np.nan

                epoch_train_loss.append(sub_loss_mean)
                epoch_train_mape.append(sub_mape_mean)
                epoch_train_rmse.append(sub_rmse_mean)

                if self.is_main_process:
                    self._log(
                        f"Epoch {epoch:03d} | Train Sub {subgraph_idx} | "
                        f"Loss: {sub_loss_mean:.4f} | MAPE: {sub_mape_mean:.4f} | RMSE: {sub_rmse_mean:.4f}"
                    )

            # ---------- VALIDATE ----------
            all_val_mae = []
            all_val_mape = []
            all_val_rmse = []

            for subgraph_idx in range(1, len(self.datasets) + 1):
                item = subgraphs[subgraph_idx]

                val_mae, val_mape, val_rmse = self.vali(
                    item["val_loader"],
                    item["scaler"],
                    item["scaler_wave"],
                    item["bathymetry"],
                    item["manning"],
                )

                all_val_mae.append(val_mae)
                all_val_mape.append(val_mape)
                all_val_rmse.append(val_rmse)

                if self.is_main_process:
                    self._log(
                        f"Epoch {epoch:03d} | Val Sub {subgraph_idx} | "
                        f"MAE: {val_mae:.4f} | MAPE: {val_mape:.4f} | RMSE: {val_rmse:.4f}"
                    )

            # ---------- EPOCH SUMMARY ----------
            avg_train_loss = float(np.nanmean(epoch_train_loss))
            avg_train_mape = float(np.nanmean(epoch_train_mape))
            avg_train_rmse = float(np.nanmean(epoch_train_rmse))

            avg_val_mae = float(np.nanmean(all_val_mae))
            avg_val_mape = float(np.nanmean(all_val_mape))
            avg_val_rmse = float(np.nanmean(all_val_rmse))

            self.lr_scheduler.step(avg_val_mae)

            epoch_time = time.time() - epoch_start
            gpu_mem_alloc = (
                torch.cuda.max_memory_allocated(self.device) / 1_000_000
                if torch.cuda.is_available() else 0.0
            )
            avg_gpu_mem = self._reduce_mean(gpu_mem_alloc)

            if self.is_main_process:
                training_history.append({
                    "Epoch": epoch,
                    "Train_Mode": self.train_mode,
                    "Val_Mode": self.val_mode,
                    "Avg_Train_Loss": avg_train_loss,
                    "Avg_Train_MAPE": avg_train_mape,
                    "Avg_Train_RMSE": avg_train_rmse,
                    "Avg_Val_MAE": avg_val_mae,
                    "Avg_Val_MAPE": avg_val_mape,
                    "Avg_Val_RMSE": avg_val_rmse,
                    "LR": self.optimizer.param_groups[0]["lr"],
                    "Time_sec": epoch_time,
                    "GPU_Mem_MB": avg_gpu_mem,
                })

                self._log(
                    f"Epoch {epoch:03d} Summary | "
                    f"Train Loss: {avg_train_loss:.4f} | Train MAPE: {avg_train_mape:.4f} | Train RMSE: {avg_train_rmse:.4f} | "
                    f"Val MAE: {avg_val_mae:.4f} | Val MAPE: {avg_val_mape:.4f} | Val RMSE: {avg_val_rmse:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {epoch_time:.1f}s | GPU Mem: {avg_gpu_mem:.1f}MB"
                )

                if avg_val_mae < min_val_mae:
                    min_val_mae = avg_val_mae
                    torch.save(self.unwrap_model().state_dict(), self.model_file)
                    self._log(f"Saved Best Model with Avg Val MAE: {min_val_mae:.4f} at Epoch {epoch:03d}")

                early_stopping(avg_val_mae)
                stop_now = early_stopping.early_stop
            else:
                stop_now = False

            if self.distributed:
                stop_tensor = torch.tensor(int(stop_now), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())

            if self.is_main_process:
                train_excel_path = os.path.join(self.log_dir, "training_results.xlsx")
                with pd.ExcelWriter(train_excel_path, mode="w", engine="openpyxl") as writer:
                    pd.DataFrame(training_history).to_excel(writer, sheet_name="Epoch_History", index=False)
                self._log(f"Training results saved to {train_excel_path}")

            if stop_now:
                break

            torch.cuda.empty_cache()

        if self.is_main_process:
            # use the last subgraph scalers only if they are shared;
            # if scalers differ by subgraph, test() should pull them per subgraph itself
            self.test(start_total_time)
            
    @torch.no_grad()
    def vali(self, val_loader, scaler, scaler_wave, bathymetry, manning):
        self.model.eval()
        valid_mae, valid_mape, valid_rmse = [], [], []

        with torch.inference_mode():
            for x, y, _ in val_loader:
                dow = x[..., -8].to(self.device, non_blocking=True)
                tod = x[..., -7].to(self.device, non_blocking=True)
                x = torch.as_tensor(x[..., self.input_feature_idx], dtype=torch.float32, device=self.device)
                y = torch.as_tensor(y[..., self.output_feature_idx], dtype=torch.float32, device=self.device)

                with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    y_hat = self._run_model(
                        x, tod, dow, bathymetry, manning,
                        mode=self.val_mode,
                    )

                y_target = y[:, :y_hat.shape[1]]

                y_hat_phys = self.inverse_transform_by_feature_group(
                    y_hat, self.output_feature_idx, scaler, scaler_wave
                )
                y_phys = self.inverse_transform_by_feature_group(
                    y_target, self.output_feature_idx, scaler, scaler_wave
                )

                valid_mae.append(masked_mae(y_hat_phys, y_phys, 0.0).item())
                valid_mape.append(masked_mape(y_hat_phys, y_phys, 0.0).item())
                valid_rmse.append(masked_rmse(y_hat_phys, y_phys, 0.0).item())

        val_mae = self._reduce_mean(np.mean(valid_mae))
        val_mape = self._reduce_mean(np.mean(valid_mape))
        val_rmse = self._reduce_mean(np.mean(valid_rmse))
        return val_mae, val_mape, val_rmse
    
    @torch.no_grad()
    def test(
        self,
        start_time,
        split: str = "test",
        save_npz: bool = True,
        load_best: bool = True,
    ):
        """
        Collect predictions/truths for:
        - one split: 'train' | 'val' | 'test' | 'all'
        Return:
        {
            "by_subgraph": {
                subgraph_idx: {
                    "preds": Tensor [B, T, N, C],
                    "trues": Tensor [B, T, N, C],
                    "metrics": {...},
                }
            },
            "combined": {
                "preds": Tensor [B, T, N_total, C] or None,
                "trues": Tensor [B, T, N_total, C] or None,
                "metrics": {...},
            }
        }
        """
        split = split.lower()
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of: 'train', 'val', 'test', 'all'")

        self._log("\n" + "=" * 20 + f" FINAL {split.upper()} MODE " + "=" * 20)

        if load_best:
            state_dict = torch.load(self.model_file, map_location=self.device)
            self.unwrap_model().load_state_dict(state_dict)

        self.model.eval()

        # ---------------------------------------------------------
        # map split -> loader keys
        # ---------------------------------------------------------
        loader_key_map = {
            "train": ["X_train_loader_wave"],
            "val": ["X_val_loader_wave"],
            "test": ["X_test_loader_wave"],
            "all": ["sub_loader"],
        }

        outputs = {
            "by_subgraph": {},
            "combined": {},
        }

        summary_rows = []
        horizon_rows = []

        # ---------------------------------------------------------
        # helper: concatenate subgraphs by node dimension
        # ---------------------------------------------------------
        def combine_subgraphs_by_node(subgraph_outputs):
            preds_list, trues_list = [], []

            for subgraph_idx in sorted(subgraph_outputs.keys()):
                pred = subgraph_outputs[subgraph_idx]["preds"]
                true = subgraph_outputs[subgraph_idx]["trues"]

                if pred is None or true is None:
                    continue

                preds_list.append(pred)
                trues_list.append(true)

            if len(preds_list) == 0:
                return None, None

            # align sample count
            min_samples = min(p.shape[0] for p in preds_list)
            preds_list = [p[:min_samples] for p in preds_list]
            trues_list = [t[:min_samples] for t in trues_list]

            # concatenate on node axis: [B,T,N,C] -> [B,T,sumN,C]
            combined_preds = torch.cat(preds_list, dim=2)
            combined_trues = torch.cat(trues_list, dim=2)
            return combined_preds, combined_trues

        # ---------------------------------------------------------
        # collect subgraph predictions
        # ---------------------------------------------------------
        for subgraph_idx in range(1, len(self.datasets) + 1):
            sub = self.datasets[subgraph_idx]
            scaler = sub["scaler"]
            scaler_wave = sub["scaler_wave"]

            bathymetry = torch.tensor(
                np.copy(sub["bathymetry_manning"][:, 0]),
                dtype=torch.float32,
                device=self.device,
            )
            manning = torch.tensor(
                np.copy(sub["bathymetry_manning"][:, 1]),
                dtype=torch.float32,
                device=self.device,
            )

            preds_parts = []
            trues_parts = []

            for loader_key in loader_key_map[split]:
                if loader_key not in sub:
                    continue

                loader = sub[loader_key]

                for x, y, _ in loader:
                    dow = x[..., -8].to(self.device, non_blocking=True)
                    tod = x[..., -7].to(self.device, non_blocking=True)

                    x = torch.as_tensor(
                        x[..., self.input_feature_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    y = torch.as_tensor(
                        y[..., self.output_feature_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        y_hat = self._run_model(
                            x, tod, dow, bathymetry, manning,
                            mode=self.test_mode,
                        )

                    y_target = y[:, :y_hat.shape[1]]

                    y_hat_phys = self.inverse_transform_by_feature_group(
                        y_hat, self.output_feature_idx, scaler, scaler_wave
                    )
                    y_phys = self.inverse_transform_by_feature_group(
                        y_target, self.output_feature_idx, scaler, scaler_wave
                    )

                    preds_parts.append(y_hat_phys.cpu())
                    trues_parts.append(y_phys.cpu())

            if len(preds_parts) == 0:
                outputs["by_subgraph"][subgraph_idx] = {
                    "preds": None,
                    "trues": None,
                    "metrics": None,
                }
                continue

            sub_pred = torch.cat(preds_parts, dim=0)   # [B,T,N,C]
            sub_true = torch.cat(trues_parts, dim=0)   # [B,T,N,C]

            sub_mae = masked_mae(sub_pred, sub_true, 0.0).item()
            sub_rmse = masked_rmse(sub_pred, sub_true, 0.0).item()
            sub_mape = masked_mape(sub_pred, sub_true, 0.0).item()

            outputs["by_subgraph"][subgraph_idx] = {
                "preds": sub_pred,
                "trues": sub_true,
                "metrics": {
                    "mae": sub_mae,
                    "rmse": sub_rmse,
                    "mape": sub_mape,
                },
            }

            self._log(
                f"Sub {subgraph_idx} Overall {split.capitalize()} ({self.test_mode}) -> "
                f"MAE: {sub_mae:.4f}, RMSE: {sub_rmse:.4f}, MAPE: {sub_mape:.4f}%"
            )

            summary_rows.append({
                "Split": split,
                "Subgraph": subgraph_idx,
                "Mode": self.test_mode,
                "MAE": sub_mae,
                "RMSE": sub_rmse,
                "MAPE": sub_mape,
            })

            max_h = min(self.horizon, sub_pred.shape[1])
            for h in range(max_h):
                h_mae = masked_mae(sub_pred[:, h], sub_true[:, h], 0.0).item()
                h_rmse = masked_rmse(sub_pred[:, h], sub_true[:, h], 0.0).item()
                h_mape = masked_mape(sub_pred[:, h], sub_true[:, h], 0.0).item()

                horizon_rows.append({
                    "Split": split,
                    "Subgraph": subgraph_idx,
                    "Mode": self.test_mode,
                    "Horizon_Step": h + 1,
                    "MAE": h_mae,
                    "RMSE": h_rmse,
                    "MAPE": h_mape,
                })

        # ---------------------------------------------------------
        # combined across subgraphs
        # ---------------------------------------------------------
        combined_preds, combined_trues = combine_subgraphs_by_node(outputs["by_subgraph"])

        if combined_preds is not None:
            combined_mae = masked_mae(combined_preds, combined_trues, 0.0).item()
            combined_rmse = masked_rmse(combined_preds, combined_trues, 0.0).item()
            combined_mape = masked_mape(combined_preds, combined_trues, 0.0).item()

            outputs["combined"] = {
                "preds": combined_preds,
                "trues": combined_trues,
                "metrics": {
                    "mae": combined_mae,
                    "rmse": combined_rmse,
                    "mape": combined_mape,
                },
            }

            self._log(
                f"Combined {split.capitalize()} ({self.test_mode}) -> "
                f"MAE: {combined_mae:.4f}, RMSE: {combined_rmse:.4f}, MAPE: {combined_mape:.4f}%"
            )

            summary_rows.append({
                "Split": split,
                "Subgraph": "combined",
                "Mode": self.test_mode,
                "MAE": combined_mae,
                "RMSE": combined_rmse,
                "MAPE": combined_mape,
            })

            max_h = min(self.horizon, combined_preds.shape[1])
            for h in range(max_h):
                h_mae = masked_mae(combined_preds[:, h], combined_trues[:, h], 0.0).item()
                h_rmse = masked_rmse(combined_preds[:, h], combined_trues[:, h], 0.0).item()
                h_mape = masked_mape(combined_preds[:, h], combined_trues[:, h], 0.0).item()

                horizon_rows.append({
                    "Split": split,
                    "Subgraph": "combined",
                    "Mode": self.test_mode,
                    "Horizon_Step": h + 1,
                    "MAE": h_mae,
                    "RMSE": h_rmse,
                    "MAPE": h_mape,
                })

        # ---------------------------------------------------------
        # save predictions
        # ---------------------------------------------------------
        if save_npz:
            save_dir = os.path.join(self.log_dir, f"predictions_{split}")
            os.makedirs(save_dir, exist_ok=True)

            # per-subgraph
            for subgraph_idx, data in outputs["by_subgraph"].items():
                if data["preds"] is None:
                    continue

                np.savez_compressed(
                    os.path.join(save_dir, f"subgraph_{subgraph_idx}_{split}.npz"),
                    preds=data["preds"].numpy(),
                    trues=data["trues"].numpy(),
                )

            # combined
            if outputs["combined"].get("preds", None) is not None:
                np.savez_compressed(
                    os.path.join(save_dir, f"combined_{split}.npz"),
                    preds=outputs["combined"]["preds"].numpy(),
                    trues=outputs["combined"]["trues"].numpy(),
                )

        # ---------------------------------------------------------
        # save metrics to excel
        # ---------------------------------------------------------
        test_excel_path = os.path.join(self.log_dir, "training_results.xlsx")
        with pd.ExcelWriter(
            test_excel_path,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            pd.DataFrame(summary_rows).to_excel(
                writer,
                sheet_name=f"{split}_summary",
                index=False,
            )
            pd.DataFrame(horizon_rows).to_excel(
                writer,
                sheet_name=f"{split}_horizon",
                index=False,
            )

        self._log(f"{split.capitalize()} metrics saved to {test_excel_path}")
        self._log(f"\nTotal Runtime: {(time.time() - start_time) / 60:.2f} minutes")

        return outputs